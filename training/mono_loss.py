"""
物理约束损失函数
用于 PE-MMNet v4 多模态裂纹预测模型

本模块实现了 4 个损失分量：
1. L_mse_density：预测裂纹密度与真实密度的均方误差
2. L_mono：物理单调性约束（裂纹密度随时间不可逆单调递增）
3. L_loc：定位损失（DIoU Loss，用于 [x, y, l, w]）
4. L_conf：置信度损失（BCE Loss，用于裂纹存在概率）

总损失：
  L_total = λ₁ * L_mse_density + λ₂ * L_mono + λ₃ * L_loc + λ₄ * L_conf

物理原理：
  - 格里菲斯断裂准则：裂纹密度随温度下降（热应力增加）单调递增
  - 损伤不可逆：裂纹一旦产生不会消失
  - 物理自洽：预测的裂纹位置和尺寸应与损伤场图像一致
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# =============================================================================
# 工具函数：DIoU / CIoU Loss
# =============================================================================

def box_iou(box1, box2, eps=1e-7):
    """
    计算两个边界框集合的 IoU

    边界框格式：[x, y, w, h]（中心坐标 + 尺寸）
    其中 x, y 是中心坐标，w, h 是宽度和高度

    Args:
        box1: 预测框，(batch, 4)，格式 [x, y, w, h]
        box2: 真实框，(batch, 4)，格式 [x, y, w, h]
        eps: 防止除零的小常数

    Returns:
        iou: IoU 值，(batch,)
    """
    # 确保宽高为正（防止 NaN）
    w1 = torch.clamp(box1[:, 2], min=eps)
    h1 = torch.clamp(box1[:, 3], min=eps)
    w2 = torch.clamp(box2[:, 2], min=eps)
    h2 = torch.clamp(box2[:, 3], min=eps)

    # 将 [x, y, w, h] 转换为 [x1, y1, x2, y2]（左上角 + 右下角）
    b1_x1, b1_x2 = box1[:, 0] - w1 / 2, box1[:, 0] + w1 / 2
    b1_y1, b1_y2 = box1[:, 1] - h1 / 2, box1[:, 1] + h1 / 2
    b2_x1, b2_x2 = box2[:, 0] - w2 / 2, box2[:, 0] + w2 / 2
    b2_y1, b2_y2 = box2[:, 1] - h2 / 2, box2[:, 1] + h2 / 2

    # 计算交集区域
    inter_x1 = torch.max(b1_x1, b2_x1)
    inter_y1 = torch.max(b1_y1, b2_y1)
    inter_x2 = torch.min(b1_x2, b2_x2)
    inter_y2 = torch.min(b1_y2, b2_y2)

    # 交集面积
    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)

    # 并集面积
    b1_area = w1 * h1
    b2_area = w2 * h2
    union_area = b1_area + b2_area - inter_area + eps

    # IoU
    iou = inter_area / union_area
    # 确保 IoU 在 [0, 1] 范围内
    iou = torch.clamp(iou, 0.0, 1.0)

    return iou


def diou_loss(pred_boxes, target_boxes, eps=1e-7):
    """
    计算 DIoU（Distance-IoU）损失

    DIoU Loss = 1 - DIoU
    DIoU = IoU - (dist^2 / diag^2)

    其中 dist 是两个框中心点之间的欧氏距离，diag 是最小外接矩形的对角线长度

    DIoU 相较于普通 IoU：
    - 考虑了中心点距离，收敛更快
    - 考虑了边界框尺寸，对不同尺度的框更友好

    Args:
        pred_boxes: 预测框，(batch, 4) -> [x, y, w, h]
        target_boxes: 真实框，(batch, 4) -> [x, y, w, h]
        eps: 防止除零的小常数

    Returns:
        loss: DIoU 损失，标量
    """
    # 确保宽高为正
    w1 = torch.clamp(pred_boxes[:, 2], min=eps)
    h1 = torch.clamp(pred_boxes[:, 3], min=eps)
    w2 = torch.clamp(target_boxes[:, 2], min=eps)
    h2 = torch.clamp(target_boxes[:, 3], min=eps)

    # 计算 IoU
    iou = box_iou(pred_boxes, target_boxes, eps)

    # 计算中心点距离
    center_dist_sq = (pred_boxes[:, 0] - target_boxes[:, 0]) ** 2 + \
                     (pred_boxes[:, 1] - target_boxes[:, 1]) ** 2

    # 计算最小外接矩形的对角线长度
    b1_x1, b1_x2 = pred_boxes[:, 0] - w1 / 2, pred_boxes[:, 0] + w1 / 2
    b1_y1, b1_y2 = pred_boxes[:, 1] - h1 / 2, pred_boxes[:, 1] + h1 / 2
    b2_x1, b2_x2 = target_boxes[:, 0] - w2 / 2, target_boxes[:, 0] + w2 / 2
    b2_y1, b2_y2 = target_boxes[:, 1] - h2 / 2, target_boxes[:, 1] + h2 / 2

    enclosing_x1 = torch.min(b1_x1, b2_x1)
    enclosing_y1 = torch.min(b1_y1, b2_y1)
    enclosing_x2 = torch.max(b1_x2, b2_x2)
    enclosing_y2 = torch.max(b1_y2, b2_y2)

    # 对角线长度平方（确保不为零）
    diag_dist_sq = torch.clamp(
        (enclosing_x2 - enclosing_x1) ** 2 + (enclosing_y2 - enclosing_y1) ** 2,
        min=eps
    )

    # DIoU = IoU - (center_dist / diag)
    diou = iou - (center_dist_sq / diag_dist_sq)
    # 确保 DIoU 在合理范围内
    diou = torch.clamp(diou, -1.0, 1.0)

    # 损失 = 1 - DIoU
    loss = 1.0 - diou

    return loss.mean()


def ciou_loss(pred_boxes, target_boxes, eps=1e-7):
    """
    计算 CIoU（Complete-IoU）损失

    CIoU = IoU - (dist^2 / diag^2) - (v^2 / (1 - IoU + v))

    其中 v 衡量宽高比的一致性：
    v = (4 / π^2) * (arctan(w_gt / h_gt) - arctan(w_pred / h_pred))^2

    CIoU 在 DIoU 基础上增加了宽高比约束，使收敛更稳定

    Args:
        pred_boxes: 预测框，(batch, 4) -> [x, y, w, h]
        target_boxes: 真实框，(batch, 4) -> [x, y, w, h]
        eps: 防止除零的小常数

    Returns:
        loss: CIoU 损失，标量
    """
    # 计算 IoU
    iou = box_iou(pred_boxes, target_boxes, eps)

    # 计算中心点距离
    center_dist_sq = (pred_boxes[:, 0] - target_boxes[:, 0]) ** 2 + \
                     (pred_boxes[:, 1] - target_boxes[:, 1]) ** 2

    # 计算外接矩形对角线
    b1_x1, b1_x2 = pred_boxes[:, 0] - pred_boxes[:, 2] / 2, pred_boxes[:, 0] + pred_boxes[:, 2] / 2
    b1_y1, b1_y2 = pred_boxes[:, 1] - pred_boxes[:, 3] / 2, pred_boxes[:, 1] + pred_boxes[:, 3] / 2
    b2_x1, b2_x2 = target_boxes[:, 0] - target_boxes[:, 2] / 2, target_boxes[:, 0] + target_boxes[:, 2] / 2
    b2_y1, b2_y2 = target_boxes[:, 1] - target_boxes[:, 3] / 2, target_boxes[:, 1] + target_boxes[:, 3] / 2

    enclosing_x1 = torch.min(b1_x1, b2_x1)
    enclosing_y1 = torch.min(b1_y1, b2_y1)
    enclosing_x2 = torch.max(b1_x2, b2_x2)
    enclosing_y2 = torch.max(b1_y2, b2_y2)

    diag_dist_sq = (enclosing_x2 - enclosing_x1) ** 2 + (enclosing_y2 - enclosing_y1) ** 2 + eps

    # 计算宽高比参数 v
    pred_w = pred_boxes[:, 2]
    pred_h = pred_boxes[:, 3]
    target_w = target_boxes[:, 2]
    target_h = target_boxes[:, 3]

    # 避免除零
    pred_w = torch.clamp(pred_w, min=eps)
    pred_h = torch.clamp(pred_h, min=eps)
    target_w = torch.clamp(target_w, min=eps)
    target_h = torch.clamp(target_h, min=eps)

    v = (4 / (torch.pi ** 2)) * ((torch.atan(target_w / target_h) - torch.atan(pred_w / pred_h)) ** 2)

    # CIoU
    ciou = iou - (center_dist_sq / diag_dist_sq) - (v ** 2 / ((1 - iou + v) + eps))

    loss = 1.0 - ciou

    return loss.mean()


# =============================================================================
# 单调性约束损失（物理核心）
# =============================================================================

class MonotonicityLossV3(nn.Module):
    """
    物理单调性约束损失（v4 版本）

    物理原理：格里菲斯断裂准则
    - 裂纹密度随温度下降（热应力增加）不可逆地单调递增
    - 损伤一旦产生不会消失

    实现方式：成对排序损失（Pairwise Ranking Loss）
    - 对于时间序列中的相邻时间步 (t_i, t_j)，其中 i < j
    - 强制 pred[j] >= pred[i]（密度只能增加不能减少）

    损失计算：
    L_mono = sum_{i<j} max(0, pred[i] - pred[j]) / num_pairs

    即：只要出现 pred[i] > pred[j]（密度下降），就计算惩罚
    """

    def __init__(self, weight=1.0, eps=1e-6):
        """
        Args:
            weight: 损失权重（乘到最终损失上）
            eps: 防止除零的小常数
        """
        super().__init__()
        self.weight = weight
        self.eps = eps

    def forward(self, pred_density, target_density=None):
        """
        计算单调性约束损失

        Args:
            pred_density: 预测的裂纹密度，(batch,) 或 (batch, 1)
            target_density: 真实裂纹密度（可选，仅用于记录，不参与计算）

        Returns:
            loss_mono: 单调性损失，标量
        """
        # 确保形状正确：(batch, 1) → (batch,)
        if pred_density.dim() == 2:
            pred_density = pred_density.squeeze(-1)

        batch_size = pred_density.size(0)

        if batch_size < 2:
            # 样本数不足，无法计算单调性，返回 0
            return self.weight * torch.tensor(0.0, device=pred_density.device)

        # 方法：检查相邻时间步的密度变化
        # 对于排好序的样本（按时间顺序），检查是否单调递增
        # 如果 pred[i] > pred[i+1]，说明出现了密度下降（违反单调性）

        # 计算相邻差分：pred[i+1] - pred[i]
        # 单调递增要求：diff >= 0
        diff = pred_density[1:] - pred_density[:-1]  # (batch-1,)

        # 惩罚负的差分（密度下降）：max(0, -diff)
        violation = torch.clamp(-diff, min=0)  # (batch-1,)

        # 损失 = 违反量的均值
        loss_mono = violation.mean()

        return self.weight * loss_mono


class SequentialMonotonicityLoss(nn.Module):
    """
    序列级单调性损失

    适用于时间序列预测场景：
    - 输入是一段时间序列（多个时间步）
    - 强制整个序列单调递增

    与 MonotonicityLossV3 的区别：
    - MonotonicityLossV3：batch 内样本间的单调性
    - SequentialMonotonicityLoss：一个样本内时间维度的单调性
    """

    def __init__(self, weight=1.0, eps=1e-6):
        super().__init__()
        self.weight = weight
        self.eps = eps

    def forward(self, pred_seq):
        """
        计算序列级单调性损失

        Args:
            pred_seq: 预测序列，(batch, seq_len)

        Returns:
            loss: 单调性损失，标量
        """
        if pred_seq.dim() == 3:
            pred_seq = pred_seq.squeeze(-1)  # (batch, seq_len)

        batch_size, seq_len = pred_seq.shape

        if seq_len < 2:
            return self.weight * torch.tensor(0.0, device=pred_seq.device)

        # 计算时间维度的差分
        # pred[:, 1:] - pred[:, :-1]：每个样本相邻时间步的差值
        diff = pred_seq[:, 1:] - pred_seq[:, :-1]  # (batch, seq_len-1)

        # 惩罚负的差分（密度下降）
        violation = torch.clamp(-diff, min=0)  # (batch, seq_len-1)

        # 取所有样本的均值
        loss_mono = violation.mean()

        return self.weight * loss_mono


# =============================================================================
# 定位损失（用于 [x, y, l, w]）
# =============================================================================

class LocalizationLoss(nn.Module):
    """
    定位损失

    结合 DIoU Loss 和 MSE Loss：
    - DIoU Loss：考虑边界框的位置和尺寸
    - MSE Loss：直接惩罚坐标偏差

    L_loc = DIoU(pred_box, target_box) + MSE(pred_coords, target_coords)
    """

    def __init__(self, lambda_diou=1.0, lambda_mse=0.5, use_ciou=False):
        """
        Args:
            lambda_diou: DIoU 损失的权重
            lambda_mse: MSE 损失的权重
            use_ciou: 是否使用 CIoU（默认 False，使用 DIoU）
        """
        super().__init__()
        self.lambda_diou = lambda_diou
        self.lambda_mse = lambda_mse
        self.use_ciou = use_ciou
        self.mse = nn.MSELoss()

    def forward(self, pred_boxes, target_boxes):
        """
        计算定位损失

        Args:
            pred_boxes: 预测框，(batch, 4) -> [x, y, w, h]
            target_boxes: 真实框，(batch, 4) -> [x, y, w, h]

        Returns:
            loss_loc: 定位损失，标量
        """
        # DIoU / CIoU 损失
        if self.use_ciou:
            loss_iou = ciou_loss(pred_boxes, target_boxes)
        else:
            loss_iou = diou_loss(pred_boxes, target_boxes)

        # MSE 损失（仅对坐标部分）
        loss_mse = self.mse(pred_boxes, target_boxes)

        # 总损失
        loss_loc = self.lambda_diou * loss_iou + self.lambda_mse * loss_mse

        return loss_loc


# =============================================================================
# 置信度损失（用于裂纹存在概率）
# =============================================================================

class ConfidenceLoss(nn.Module):
    """
    置信度损失

    使用二元交叉熵（BCE）损失：
    - 真实标签规则：如果真实裂纹密度 > 0.5，则为 1（存在裂纹），否则为 0（无裂纹）
    - 预测值：模型输出的置信度 (0~1)

    设计选择：
    - 使用 BCEWithLogitsLoss（Sigmoid + BCE 合并），数值更稳定
    - 也可以使用普通 BCE + Sigmoid
    """

    def __init__(self, pos_weight=1.0):
        """
        Args:
            pos_weight: 正样本权重（用于处理类别不平衡）
        """
        super().__init__()
        # pos_weight > 1 会增加正样本的惩罚
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))

    def forward(self, pred_confidence, target_confidence):
        """
        计算置信度损失

        Args:
            pred_confidence: 预测置信度，(batch,) 或 (batch, 1)，值域 [0, 1]
            target_confidence: 真实置信度，(batch,) 或 (batch, 1)，值为 0 或 1

        Returns:
            loss_conf: 置信度损失，标量
        """
        # 确保形状正确
        if pred_confidence.dim() == 2:
            pred_confidence = pred_confidence.squeeze(-1)
        if target_confidence.dim() == 2:
            target_confidence = target_confidence.squeeze(-1)

        # 转换为 logits（Sigmoid 的逆操作）
        # BCEWithLogitsLoss 内部会做 Sigmoid，所以输入应该是 logit
        eps = 1e-3  # 增大 eps 防止极端值
        pred_confidence = torch.clamp(pred_confidence, eps, 1 - eps)

        # 安全计算 logits
        logits = torch.log(pred_confidence / (1 - pred_confidence))
        logits = torch.clamp(logits, -50, 50)  # 限制 logits 范围

        loss_conf = self.criterion(logits, target_confidence)

        # 检查 NaN
        if torch.isnan(loss_conf):
            return torch.tensor(0.1, device=pred_confidence.device, requires_grad=True)

        return loss_conf


# =============================================================================
# 组合损失函数
# =============================================================================

class MultimodalCrackLoss(nn.Module):
    """
    多模态裂纹预测组合损失函数

    L_total = λ₁ * L_mse_density + λ₂ * L_mono + λ₃ * L_loc + λ₄ * L_conf

    各损失分量：
    1. L_mse_density：预测密度与真实密度的 MSE
    2. L_mono：物理单调性约束（裂纹密度随时间单调递增）
    3. L_loc：定位损失（DIoU + MSE for [x, y, l, w]）
    4. L_conf：置信度损失（BCE for 裂纹存在概率）

    标签格式（6 维）：
    [x, y, l, w, confidence, density]
    """

    def __init__(self,
                 lambda_mse_density=1.0,
                 lambda_mono=0.1,
                 lambda_loc=1.0,
                 lambda_conf=1.0,
                 use_ciou=False):
        """
        Args:
            lambda_mse_density: MSE 损失权重
            lambda_mono: 单调性损失权重
            lambda_loc: 定位损失权重
            lambda_conf: 置信度损失权重
            use_ciou: 是否使用 CIoU（默认 False）
        """
        super().__init__()

        self.lambda_mse_density = lambda_mse_density
        self.lambda_mono = lambda_mono
        self.lambda_loc = lambda_loc
        self.lambda_conf = lambda_conf

        # 各损失分量
        self.mse_density = nn.MSELoss()
        self.monotonicity = MonotonicityLossV3(weight=1.0)
        self.localization = LocalizationLoss(use_ciou=use_ciou)
        self.confidence = ConfidenceLoss()

    def forward(self, pred, target):
        """
        计算总损失

        Args:
            pred: 预测输出，(batch, 6) -> [x, y, l, w, confidence, density]
            target: 真实标签，(batch, 6) -> [x, y, l, w, confidence, density]

        Returns:
            loss_total: 总损失，标量
            loss_dict: 各损失分量字典
        """
        # -------------------------------------------------------------------------
        # 拆分预测和标签
        # -------------------------------------------------------------------------
        # 预测
        pred_xy = pred[:, 0:2]      # (batch, 2)
        pred_lw = pred[:, 2:4]      # (batch, 2)
        pred_conf = pred[:, 4:5]    # (batch, 1)
        pred_density = pred[:, 5:6] # (batch, 1)

        # 标签
        target_xy = target[:, 0:2]
        target_lw = target[:, 2:4]
        target_conf = target[:, 4:5]
        target_density = target[:, 5:6]

        # -------------------------------------------------------------------------
        # 计算各损失分量
        # -------------------------------------------------------------------------
        # 1. MSE 密度损失
        loss_mse = self.mse_density(pred_density, target_density)

        # 2. 单调性损失（仅对密度）
        loss_mono = self.monotonicity(pred_density, target_density)

        # 3. 定位损失（对 [x, y, l, w]）
        # 拼接为边界框格式：[x, y, w, h]
        pred_box = torch.cat([pred_xy, pred_lw], dim=1)
        target_box = torch.cat([target_xy, target_lw], dim=1)
        loss_loc = self.localization(pred_box, target_box)

        # 4. 置信度损失
        loss_conf = self.confidence(pred_conf, target_conf)

        # -------------------------------------------------------------------------
        # 计算总损失（防止 NaN）
        # -------------------------------------------------------------------------
        # 安全计算：确保每个分量都是有效值
        loss_mse = torch.where(torch.isnan(loss_mse), torch.zeros_like(loss_mse), loss_mse)
        loss_mono = torch.where(torch.isnan(loss_mono), torch.zeros_like(loss_mono), loss_mono)
        loss_loc = torch.where(torch.isnan(loss_loc), torch.zeros_like(loss_loc), loss_loc)
        loss_conf = torch.where(torch.isnan(loss_conf), torch.zeros_like(loss_conf), loss_conf)

        loss_total = (
            self.lambda_mse_density * loss_mse +
            self.lambda_mono * loss_mono +
            self.lambda_loc * loss_loc +
            self.lambda_conf * loss_conf
        )

        # 最终 NaN 检查
        if torch.isnan(loss_total):
            loss_total = loss_mse + loss_loc  # 回退到基础损失

        # -------------------------------------------------------------------------
        # 组装损失字典
        # -------------------------------------------------------------------------
        loss_dict = {
            'mse_density': loss_mse.item(),
            'mono': loss_mono.item(),
            'loc': loss_loc.item(),
            'conf': loss_conf.item(),
            'total': loss_total.item()
        }

        return loss_total, loss_dict


# =============================================================================
# 辅助函数：验证单调性
# =============================================================================

def verify_monotonicity(predictions, times=None):
    """
    验证预测结果是否满足单调性约束

    Args:
        predictions: 预测的裂纹密度序列，(N,) 或 (N, 1)
        times: 对应的时间点（可选）

    Returns:
        dict: 包含单调性验证结果
    """
    if predictions.dim() == 2:
        predictions = predictions.squeeze(-1)

    predictions = predictions.cpu().numpy()

    # 检查相邻差分
    diffs = predictions[1:] - predictions[:-1]

    # 统计违反次数
    violations = np.sum(diffs < 0)

    # 计算单调性得分（越高越好）
    monotonicity_score = 1.0 - violations / max(len(diffs), 1)

    return {
        'violations': int(violations),
        'total_pairs': len(diffs),
        'monotonicity_score': float(monotonicity_score),
        'max_decrease': float(np.min(diffs)) if violations > 0 else 0.0,
        'max_increase': float(np.max(diffs))
    }


# =============================================================================
# 分割损失函数
# =============================================================================

class DiceLoss(nn.Module):
    """
    Dice Loss 用于图像分割任务

    原理：
    - Dice = 2 * |A ∩ B| / (|A| + |B|)
    - 最大化 Dice 等价于最小化 1 - Dice
    - 对类别不平衡更鲁棒

    适用于裂纹掩膜分割，裂纹区域通常只占图像的一小部分
    """

    def __init__(self, smooth=1.0):
        """
        Args:
            smooth: 平滑项，防止除零
        """
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        """
        计算 Dice Loss

        Args:
            pred: 预测掩膜，(batch, 1, H, W) 或 (batch, H, W)，值域 [0, 1]
            target: 真实掩膜，同上

        Returns:
            loss: Dice Loss，标量
        """
        # 确保形状一致
        pred = pred.view(-1)
        target = target.view(-1)

        # 计算交集
        intersection = (pred * target).sum()

        # Dice 系数
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)

        # 返回 Dice Loss
        return 1 - dice


class SegmentationLoss(nn.Module):
    """
    分割任务损失：Dice Loss + BCE Loss 组合

    - Dice Loss：对类别不平衡鲁棒
    - BCE Loss：像素级精确度
    """

    def __init__(self, lambda_dice=1.0, lambda_bce=0.5):
        """
        Args:
            lambda_dice: Dice Loss 权重
            lambda_bce: BCE Loss 权重
        """
        super().__init__()
        self.dice_loss = DiceLoss()
        self.bce_loss = nn.BCELoss()
        self.lambda_dice = lambda_dice
        self.lambda_bce = lambda_bce

    def forward(self, pred, target):
        """
        Args:
            pred: 预测掩膜，(batch, 1, H, W)，值域 [0, 1]
            target: 真实掩膜，(batch, 1, H, W)，值域 [0, 1]

        Returns:
            loss: 分割损失，标量
        """
        # 分割 BCE 也需要安全处理（pred 来自 MaskDecoder 已 sigmoid）
        pred_safe = torch.clamp(pred, min=1e-7, max=1-1e-7)
        loss_dice = self.dice_loss(pred, target)
        loss_bce = F.binary_cross_entropy(pred_safe, target)
        return self.lambda_dice * loss_dice + self.lambda_bce * loss_bce


class MultimodalSegmentationLoss(nn.Module):
    """
    多任务损失：检测 + 分割

    支持检测标签为 None 的情况（长序列数据只有分割标签）
    自动退化为纯分割损失
    """

    def __init__(self, lambda_seg=1.0, lambda_det=0.5):
        """
        Args:
            lambda_seg: 分割损失权重
            lambda_det: 检测损失权重
        """
        super().__init__()
        self.segmentation_loss = SegmentationLoss()
        self.detection_loss = MultimodalCrackLoss(
            lambda_mse_density=1.0,
            lambda_mono=0.1,
            lambda_loc=1.0,
            lambda_conf=1.0
        )
        self.lambda_seg = lambda_seg
        self.lambda_det = lambda_det

    def forward(self, outputs, targets):
        """
        计算多任务损失

        Args:
            outputs: (pred_mask, pred_det) 元组
            targets: (target_mask, target_det) 元组
                    target_det 可以为 None（长序列数据无检测标签）

        Returns:
            loss: 总损失，标量
        """
        pred_mask, pred_det = outputs
        target_mask, target_det = targets

        # 分割损失（必须有）
        loss_seg = self.segmentation_loss(pred_mask, target_mask)

        # 检测损失（可选）
        if target_det is not None:
            loss_det, _ = self.detection_loss(pred_det, target_det)
            return self.lambda_seg * loss_seg + self.lambda_det * loss_det
        else:
            # 只有分割标签，退化为纯分割
            return loss_seg


# =============================================================================
# YOLO 损失函数（用于 Swin-YOLO-FPN 和 ViT-YOLO-FPN）
# =============================================================================

class YOLOTargetAssigner(nn.Module):
    """
    YOLO 目标分配器

    将真实目标分配到 YOLO 网格：
    - 中心网格：完整目标标签
    - 附近网格（可选）：降低置信度的目标标签

    Args:
        grid_size: 网格尺寸（默认 16 -> 256 个网格）
        nearby_range: 附近网格分配范围（默认 2 -> 5x5 区域）
    """

    def __init__(self, grid_size=16, nearby_range=2):
        super().__init__()
        self.grid_size = grid_size
        self.nearby_range = nearby_range

    def forward(self, targets):
        """
        分配目标到网格

        Args:
            targets: (B, 6) 真实标签 [x, y, l, w, conf, density]

        Returns:
            assigned_targets: (B, num_grids, 6) 每个网格的目标标签
            positive_mask: (B, num_grids) 正样本掩码
        """
        B = targets.size(0)
        device = targets.device
        num_grids = self.grid_size * self.grid_size

        # 初始化
        assigned = torch.zeros(B, num_grids, 6, device=device)
        positive_mask = torch.zeros(B, num_grids, dtype=torch.bool, device=device)

        for b in range(B):
            # 提取目标坐标
            x, y = targets[b, 0].item(), targets[b, 1].item()
            conf = targets[b, 4].item()
            density = targets[b, 5].item()

            # 计算中心网格索引
            cell_x = min(int(x * self.grid_size), self.grid_size - 1)
            cell_y = min(int(y * self.grid_size), self.grid_size - 1)
            center_idx = cell_y * self.grid_size + cell_x

            # 分配到中心网格
            assigned[b, center_idx] = targets[b]
            positive_mask[b, center_idx] = True

            # 分配到附近网格（增强正样本）
            for dy in range(-self.nearby_range, self.nearby_range + 1):
                for dx in range(-self.nearby_range, self.nearby_range + 1):
                    if dx == 0 and dy == 0:
                        continue

                    nx, ny = cell_x + dx, cell_y + dy

                    # 检查是否在边界内
                    if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                        idx = ny * self.grid_size + nx

                        # 附近网格使用相同目标，但降低置信度
                        nearby_target = targets[b].clone()
                        nearby_target[4] = conf * 0.8  # 降低置信度
                        assigned[b, idx] = nearby_target
                        positive_mask[b, idx] = True

        return assigned, positive_mask


class YOLOLoss(nn.Module):
    """
    YOLO 检测损失

    包含：
    - 定位损失 (DIoU)
    - 置信度损失 (BCE)
    - 密度损失 (MSE)
    - 单调性损失（简化版）

    Args:
        lambda_box: 定位损失权重
        lambda_conf: 置信度损失权重
        lambda_mono: 单调性损失权重
    """

    def __init__(self, lambda_box=1.0, lambda_conf=1.0, lambda_mono=0.1):
        super().__init__()
        self.lambda_box = lambda_box
        self.lambda_conf = lambda_conf
        self.lambda_mono = lambda_mono

    def forward(self, pred, target, global_density, target_density, positive_mask=None):
        """
        计算 YOLO 损失

        Args:
            pred: (B, num_grids, 6) 网格预测
            target: (B, num_grids, 6) 分配后的目标
            global_density: (B, 1) 最大密度（用于单调性）
            target_density: (B, 1) 目标密度
            positive_mask: (B, num_grids) 正样本掩码

        Returns:
            loss: 总损失
        """
        # 如果没有提供正样本掩码，使用默认策略
        if positive_mask is None:
            positive_mask = (target[..., 4:5].sum(dim=1) > 0).squeeze(-1)

        # ========== 定位损失 (DIoU) ==========
        pred_boxes = pred[..., :4][positive_mask]
        target_boxes = target[..., :4][positive_mask]

        if len(pred_boxes) > 0:
            loss_box = diou_loss(pred_boxes, target_boxes)
        else:
            loss_box = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

        # ========== 置信度损失 ==========
        pred_conf = pred[..., 4:5][positive_mask]
        target_conf = target[..., 4:5][positive_mask]

        if len(pred_conf) > 0:
            # 使用 binary_cross_entropy_with_logits（安全用于 FP16）
            # 但输入应该是 logits，需要 clip 避免 infinity
            pred_conf_logits = torch.clamp(pred_conf, min=-50, max=50)
            loss_conf = F.binary_cross_entropy_with_logits(
                pred_conf_logits, target_conf
            )
        else:
            loss_conf = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

        # ========== 密度 MSE 损失 ==========
        # 防御性：确保 global_density 和 target_density 维度一致
        if global_density.ndim == 3:
            global_density = global_density.squeeze(-1)  # (B, 1, 1) -> (B, 1)
        loss_density = F.mse_loss(global_density, target_density)

        # ========== 单调性损失（简化版） ==========
        # 对于 YOLO/DETR，密度是单值输出，简化为与目标密度的差异
        loss_mono = torch.abs(global_density - target_density).mean()

        # ========== 总损失 ==========
        loss_total = (
            self.lambda_box * loss_box +
            self.lambda_conf * loss_conf +
            self.lambda_mono * loss_mono +
            loss_density
        )

        return loss_total


# =============================================================================
# DETR 损失函数
# =============================================================================

class HungarianMatcher(nn.Module):
    """
    Hungarian Matcher for DETR

    使用 scipy.optimize.linear_sum_assignment 找到预测与目标的最佳匹配

    Args:
        cost_bbox: 边界框成本权重
        cost_density: 密度成本权重
    """

    def __init__(self, cost_bbox=1.0, cost_density=1.0):
        super().__init__()
        self.cost_bbox = cost_bbox
        self.cost_density = cost_density

    @torch.no_grad()
    def forward(self, outputs, targets):
        """
        Hungarian 匹配

        Args:
            outputs: (B, num_queries, 6) 预测 [x, y, l, w, conf, density]
            targets: dict {'labels': (B, 6)} 真实标签

        Returns:
            indices: list of (pred_idx, target_idx) pairs per batch
        """
        from scipy.optimize import linear_sum_assignment

        B, num_queries = outputs.shape[:2]
        device = outputs.device

        # 提取预测和目标
        out_bbox = outputs[..., :4]  # (B, num_queries, 4)
        out_density = outputs[..., 5:6]  # (B, num_queries, 1)

        tgt_labels = targets['labels']
        # DETR 检测场景：每个样本有 1 个目标（6维向量）
        # 注意：tgt_labels.shape[1]=6 是特征维度，不是目标数量
        tgt_bbox = tgt_labels[..., :4].unsqueeze(1)  # (B, 1, 4)
        tgt_density = tgt_labels[..., 5:6].unsqueeze(1)  # (B, 1, 1)

        # 构建成本矩阵
        # 边界框成本：L1 距离
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)  # (B, num_queries, num_targets)

        # 密度成本：L1 距离
        cost_density = torch.abs(out_density - tgt_density)  # (B, num_queries, num_targets)

        # 总成本
        cost_matrix = self.cost_bbox * cost_bbox + self.cost_density * cost_density
        # 移除目标维度（假设每个样本只有一个目标）
        cost_matrix = cost_matrix.squeeze(-1)  # (B, num_queries)

        # Hungarian 匹配
        indices = []
        for b in range(B):
            # scipy 需要 2D 矩阵 (num_queries x num_targets)
            cost_b = cost_matrix[b].cpu().numpy()
            # 确保是 2D
            if cost_b.ndim == 1:
                cost_b = cost_b.reshape(-1, 1)
            row_ind, col_ind = linear_sum_assignment(cost_b)

            # 转换为 PyTorch 张量
            indices.append(torch.tensor(
                list(zip(row_ind, col_ind)),
                dtype=torch.long,
                device=device
            ))

        return indices


class DETRLoss(nn.Module):
    """
    DETR 检测损失

    包含：
    - 定位损失 (Smooth L1)
    - 置信度损失 (BCE)
    - 密度损失 (MSE)
    - 单调性损失（简化版）

    与 YOLOLoss 的区别：
    - 使用 Hungarian Matching 进行预测-目标匹配
    - 每个样本可能有不同的匹配对

    Args:
        matcher: HungarianMatcher 实例
        lambda_bbox: 定位损失权重
        lambda_conf: 置信度损失权重
        lambda_mono: 单调性损失权重
    """

    def __init__(self, matcher=None, lambda_bbox=1.0, lambda_conf=1.0, lambda_mono=0.1):
        super().__init__()

        if matcher is None:
            matcher = HungarianMatcher()

        self.matcher = matcher
        self.lambda_bbox = lambda_bbox
        self.lambda_conf = lambda_conf
        self.lambda_mono = lambda_mono

    def forward(self, pred, target, global_density, target_density, indices=None, final_output=None):
        """
        计算 DETR 损失

        Args:
            pred: (B, num_queries, 6) query 预测
            target: dict {'labels': (B, 6)} 真实标签
            global_density: (B, 1) 最大密度
            target_density: (B, 1) 目标密度
            indices: Hungarian 匹配结果（可选）
            final_output: (B, 6) 融合后的输出（query_proj→fusion→output_head，用于监督损失）

        Returns:
            loss: 总损失
        """
        # 获取匹配结果
        if indices is None:
            indices = self.matcher(pred, target)

        # 初始化损失
        loss_bbox = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        loss_conf = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

        # 计算匹配的损失
        for b, idx_per_batch in enumerate(indices):
            if len(idx_per_batch) == 0:
                continue

            pred_idx, tgt_idx = idx_per_batch[:, 0], idx_per_batch[:, 1]

            # target['labels'] 是 (B, 6)，tgt_idx 是匹配的 query 索引
            # 我们用 pred[b, pred_idx] 与 target['labels'][b] 进行比较
            tgt_labels = target['labels'][b]  # (6,)

            # 定位损失 (Smooth L1) - 比较预测的 query 与目标
            loss_bbox += F.smooth_l1_loss(
                pred[b, pred_idx, :4],
                tgt_labels[:4].unsqueeze(0).expand(len(pred_idx), -1)
            ).mean()

            # 置信度损失 (BCE) - 使用 binary_cross_entropy_with_logits 安全用于 FP16
            loss_conf += F.binary_cross_entropy_with_logits(
                torch.clamp(pred[b, pred_idx, 4:5], min=-50, max=50),
                tgt_labels[4:5].unsqueeze(0).expand(len(pred_idx), -1)
            ).mean()

        # 平均
        B = pred.size(0)
        loss_bbox = loss_bbox / B
        loss_conf = loss_conf / B

        # 密度和单调性损失
        # global_density: (B,1,1) from model max(), target_density: (B,1)
        if global_density.ndim == 3:
            global_density = global_density.squeeze(-1)  # (B,1,1) → (B,1)
        loss_density = F.mse_loss(global_density, target_density)
        loss_mono = torch.abs(global_density - target_density).mean()

        # 总损失
        loss_total = (
            self.lambda_bbox * loss_bbox +
            self.lambda_conf * loss_conf +
            self.lambda_mono * loss_mono +
            loss_density
        )

        # 监督融合层输出（训练/推理特征空间一致的关键）
        if final_output is not None:
            tgt_labels = target['labels']  # (B, 6)
            loss_supervise = F.smooth_l1_loss(
                final_output[:, :4], tgt_labels[:, :4]
            ) + F.mse_loss(
                final_output[:, 5:6], tgt_labels[:, 5:6]
            )
            loss_total = loss_total + 0.5 * loss_supervise

        return loss_total


# =============================================================================
# 密度一致性损失
# =============================================================================

class DensityConsistencyLoss(nn.Module):
    """
    密度分布一致性损失

    物理原理：
    - 相邻网格的裂纹密度应该相近（空间平滑性）
    - 裂纹是连续结构，相邻位置的密度不会有突变

    实现：
    - 对于每个正样本网格，计算其与3x3邻域内其他正样本网格的密度差异
    - 邻域密度差异越小，一致性越好

    使用方式：
    ```python
    density_loss = DensityConsistencyLoss(grid_size=16, neighbor_range=1)
    loss = density_loss(grid_pred, positive_mask)
    ```

    Args:
        grid_size: 网格尺寸（默认16，即16x16=256个网格）
        neighbor_range: 邻域范围（默认1，即3x3邻域）
        lambda_consistency: 一致性损失权重
    """

    def __init__(self, grid_size=16, neighbor_range=1, lambda_consistency=0.5):
        super().__init__()
        self.grid_size = grid_size
        self.neighbor_range = neighbor_range
        self.lambda_consistency = lambda_consistency

    def forward(self, grid_pred, positive_mask):
        """
        计算密度一致性损失

        Args:
            grid_pred: 网格预测 (B, num_grids, 6)
                      预测向量格式: [x, y, l, w, conf, density]
                      我们只使用第5维（density）
            positive_mask: 正样本掩码 (B, num_grids)
                          True表示该网格包含真实目标

        Returns:
            一致性损失标量
        """
        # 提取密度预测
        density = grid_pred[..., 5]  # (B, num_grids)

        neighbor_loss = 0.0
        count = 0

        for b in range(density.size(0)):
            # 获取当前批次中所有正样本的索引
            pos_indices = positive_mask[b].nonzero(as_tuple=True)[0]

            for idx in pos_indices:
                # 将扁平索引转换为2D坐标
                i, j = idx // self.grid_size, idx % self.grid_size

                # 收集邻域内的正样本密度值
                neighbors = []
                for di in range(-self.neighbor_range, self.neighbor_range + 1):
                    for dj in range(-self.neighbor_range, self.neighbor_range + 1):
                        if di == 0 and dj == 0:
                            continue  # 跳过自身

                        ni, nj = i + di, j + dj
                        # 检查边界
                        if 0 <= ni < self.grid_size and 0 <= nj < self.grid_size:
                            n_idx = ni * self.grid_size + nj
                            # 检查邻域是否也是正样本
                            if positive_mask[b, n_idx]:
                                neighbors.append(density[b, n_idx])

                # 计算与邻域的一致性损失
                if neighbors:
                    center_density = density[b, idx].expand(len(neighbors))
                    neighbor_loss += F.mse_loss(
                        center_density,
                        torch.stack(neighbors)
                    )
                    count += 1

        # 平均损失
        if count > 0:
            neighbor_loss = neighbor_loss / count

        return self.lambda_consistency * neighbor_loss


class CombinedDensityLoss(nn.Module):
    """
    组合密度损失

    结合MSE密度损失和一致性损失，用于YOLO/DETR变体。

    总损失：
    L_density = λ_mse * L_mse + λ_consistency * L_consistency
    """

    def __init__(self, grid_size=16, neighbor_range=1,
                 lambda_mse=1.0, lambda_consistency=0.5):
        super().__init__()
        self.lambda_mse = lambda_mse
        self.lambda_consistency = lambda_consistency
        self.consistency_loss = DensityConsistencyLoss(
            grid_size=grid_size,
            neighbor_range=neighbor_range,
            lambda_consistency=1.0  # 内部权重设为1，在forward中乘以外部权重
        )

    def forward(self, grid_pred, target, positive_mask):
        """
        计算组合密度损失

        Args:
            grid_pred: 预测 (B, num_grids, 6)
            target: 目标 (B, num_grids, 6) - 分配后的目标
            positive_mask: 正样本掩码 (B, num_grids)

        Returns:
            总损失
        """
        # MSE密度损失（仅在正样本上计算）
        pred_density = grid_pred[..., 5]  # (B, num_grids)
        target_density = target[..., 5]  # (B, num_grids)

        # 只在正样本上计算MSE
        if positive_mask.any():
            loss_mse = F.mse_loss(
                pred_density[positive_mask],
                target_density[positive_mask]
            )
        else:
            loss_mse = torch.tensor(0.0, device=grid_pred.device)

        # 一致性损失
        loss_consistency = self.consistency_loss(grid_pred, positive_mask)

        return self.lambda_mse * loss_mse + self.lambda_consistency * loss_consistency

