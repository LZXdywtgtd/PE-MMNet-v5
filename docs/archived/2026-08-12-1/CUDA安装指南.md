# CUDA 安装指南

> GPU 加速训练配置

---

## 一、环境要求

- NVIDIA 显卡（支持 CUDA）
- 驱动版本 ≥ 450.80（支持 CUDA 11.x）

---

## 二、验证 GPU

```powershell
nvidia-smi
```

**预期输出**：
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 525.xx    Driver Version: 525.xx    CUDA Version: 12.0        |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
|   0  NVIDIA GeForce ...  Off  | 00000000:01:00.0  On |                  0 MiB |
+-----------------------------------------------------------------------------+
```

---

## 三、安装 CUDA Toolkit

### 3.1 下载

访问 [NVIDIA CUDA 下载](https://developer.nvidia.com/cuda-downloads)，选择：
- Windows > x86_64 > 11 > exe(local)

### 3.2 安装

1. 运行安装程序
2. 选择 **自定义安装**
3. 取消勾选 **GeForce Experience**（不需要）
4. 完成安装

### 3.3 验证

```powershell
nvcc --version
```

**预期输出**：
```
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2023 NVIDIA Corporation
Built on ...
Cuda compilation tools, release 11.x, V11.x.x
```

---

## 四、安装 PyTorch GPU 版本

### 4.1 访问 PyTorch 官网

访问 [pytorch.org](https://pytorch.org/)，使用官方命令生成器：

1. 选择 PyTorch 版本（如 2.0.1）
2. 选择 CUDA 版本（如 11.8）
3. 复制生成的命令

### 4.2 常用安装命令

**CUDA 11.8**：
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**CUDA 12.1**：
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 4.3 CPU 版本（无 GPU）

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

---

## 五、验证安装

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
```

**预期输出**：
```
PyTorch: 2.0.1
CUDA: True
Device: NVIDIA GeForce RTX 3080
```

---

## 六、常见问题

### Q1: `nvidia-smi` 不是内部命令

**原因**：未安装 NVIDIA 驱动

**解决**：安装 [NVIDIA GeForce Experience](https://www.nvidia.com/geforce/geforce-experience/) 或单独下载驱动

### Q2: `torch.cuda.is_available()` 返回 False

**可能原因**：
1. 未安装 GPU 版 PyTorch
2. CUDA 版本不匹配
3. 驱动版本过低

**解决**：
```bash
# 检查驱动支持的 CUDA 版本
nvidia-smi

# 检查已安装的 CUDA 编译器
nvcc --version

# 重新安装匹配的 PyTorch
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Q3: 训练速度很慢

**检查**：
1. 是否使用 GPU：`torch.cuda.is_available()` 应为 True
2. 显存占用：`nvidia-smi` 查看 GPU-Util
3. 数据加载是否在 GPU 上

---

## 七、驱动与 CUDA 版本对应

| 驱动版本 | 最高支持 CUDA |
|----------|---------------|
| ≥ 525 | CUDA 12.x |
| 450 ~ 524 | CUDA 11.x |
| < 450 | 需更新驱动 |

---

## 八、不想用 GPU？

没问题，代码支持 CPU 训练，只是速度较慢（可能慢 10-50 倍）。

安装 CPU 版本后，代码会自动检测并使用 CPU，无需修改任何设置。
