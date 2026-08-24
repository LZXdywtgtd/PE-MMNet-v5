# 飞书通知配置指南（v5）

> v5 fork 自 v4，已将默认仓库从 `PE-MMNet-v4` 改为 `PE-MMNet-v5`。
> 配置步骤与 v4 一致。

## 功能

Push 代码或发布 Release 时，飞书群会自动收到卡片通知：

```
┌─────────────────────────────────────┐
│  📝 代码更新 - master               │
├─────────────────────────────────────┤
│  feat: 添加新功能                   │
│                                     │
│  📌 提交: abc1234                   │
│  👤 作者: xxx                       │
│  🌿 分支: master                    │
├─────────────────────────────────────┤
│  📦 PE-MMNet-v5 | 查看代码 | Release│
└─────────────────────────────────────┘
```

## 配置步骤

### 1. 创建飞书机器人

1. 在飞书群中点击「设置」→「群机器人」→「添加机器人」→「自定义机器人」
2. 设置名称（如 `GitHub通知`）并添加
3. 复制 webhook 地址，格式如下：
   ```
   https://open.feishu.cn/open-apis/bot/v2/hook/xxxxxxxxxxxxxxxxx
   ```
4. **建议设置关键词**（如 `GitHub`），发送消息时必须包含该关键词

### 2. 添加 GitHub Secrets

1. 打开 GitHub 仓库页面
2. 进入 **Settings** → **Secrets and variables** → **Actions**
3. 点击 **New repository secret**
4. 创建以下 secret：

| Name | Value |
|------|-------|
| `FEISHU_WEBHOOK` | 飞书机器人的完整 webhook 地址 |

### 3. 验证

推送代码到 master/main 分支或发布新 Release，飞书群应该会收到通知。

## 自定义

### 修改关键词

如果设置了关键词，需要在 GitHub Actions 的脚本中包含该关键词。修改 `.github/workflows/feishu-notify.yml`：

```yaml
# 在发送的消息中添加关键词
content: `【GitHub】\n\n${content}`
```

### 修改卡片样式

可以修改 `card` 结构体来调整卡片颜色、布局等。
