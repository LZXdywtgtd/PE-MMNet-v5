# CUDA 安装指南

> PE-MMNet v4 GPU 环境配置 | 目标：让 `torch.cuda.is_available()` 返回 `True`

---

## 一、环境要求

| 组件 | 最低 | 推荐 |
|------|------|------|
| NVIDIA 驱动 | 450.80+（CUDA 11.x） | 525+（CUDA 12.x） |
| GPU 显存 | 4 GB | 8 GB+ |
| 操作系统 | Windows 10 / Ubuntu 18.04+ | Windows 11 / Ubuntu 22.04 |
| 磁盘 | 10 GB（CUDA Toolkit + PyTorch） | 20 GB+ |

> **无 GPU？** 跳到 §四.3 装 CPU 版 PyTorch，所有功能仍可跑（仅训练慢 10-50 倍）。

---

## 二、验证 GPU

```bash
nvidia-smi
```

预期输出包含 GPU 型号、驱动版本、CUDA 版本：

```
+-------------------------------------------------------------------------+
| NVIDIA-SMI 535.xx       Driver Version: 535.xx    CUDA Version: 12.2    |
|----------------------------+---------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. |
| 0  NVIDIA GeForce RTX 3060    On                 | 00000000:01:00.0  On |
|----------------------------+---------------------+----------------------+
```

如果命令不存在或报错，说明驱动未装；去 https://www.nvidia.com/drivers 下载对应型号。

---

## 三、安装 CUDA Toolkit

### 3.1 下载

- 官方下载：https://developer.nvidia.com/cuda-downloads
- 选择对应的 OS / 架构 / 版本（推荐 CUDA 11.8 或 12.1，与 PyTorch 兼容）

### 3.2 安装

按官方 installer 提示安装（Linux 上 `.run` 或 `.deb`；Windows 上 `.exe`）。

### 3.3 验证

```bash
nvcc --version
```

预期输出 `release 11.8` 或 `release 12.1` 等。

> **重要**：CUDA Toolkit 版本必须 ≥ PyTorch 编译时用的 CUDA 版本。

---

## 四、安装 PyTorch GPU 版本

### 4.1 访问 PyTorch 官网

去 https://pytorch.org/get-started/locally/ 选择：
- PyTorch Build：**Stable**
- Your OS：Windows / Linux
- Package：**Conda**（推荐）或 **Pip**
- Language：Python
- Compute Platform：**CUDA 11.8** / **CUDA 12.1** / **CUDA 12.4**

### 4.2 常用安装命令

**CUDA 11.8（兼容性好，推荐）**：
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**CUDA 12.1**：
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**CUDA 12.4**：
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

### 4.3 CPU 版本（无 GPU）

```bash
pip install torch torchvision
```

> CPU 版仍可运行 `team_train.py --auto` 和小规模实验（`--epochs 2` 验证用）。生产训练（150 epoch）会非常慢。

---

## 五、验证安装

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

预期输出（GPU 机器）：
```
PyTorch: 2.4.0
CUDA available: True
GPU: NVIDIA GeForce RTX 3060
```

---

## 六、常见问题

### Q1: `nvidia-smi` 找不到

**原因**：驱动未装或不在 PATH。

**解决**：
- Windows：从 NVIDIA 官网下载驱动并安装
- Linux：`sudo apt install nvidia-driver-535`（版本号按需调整）

### Q2: PyTorch 装好后 `torch.cuda.is_available()` 返回 `False`

**原因 1**：PyTorch 装的是 CPU 版。
- 解决：用 §四.2 的命令重装 GPU 版

**原因 2**：CUDA Toolkit 与 PyTorch 的 CUDA 版本不匹配。
- 解决：`nvcc --version` 看 CUDA 版本，再装对应 PyTorch（cu118/cu121/cu124）

**原因 3**：环境变量 `CUDA_HOME` 未设。
- 解决：
  ```bash
  # Linux
  export CUDA_HOME=/usr/local/cuda
  export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
  # Windows
  set CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1
  ```

### Q3: `ImportError: DLL load failed`（Windows）

**原因**：缺 Visual C++ Redistributable。

**解决**：装 [Microsoft Visual C++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe)。

### Q4: 装好后跑 `python run_train.py` 仍报 CUDA 错

跑 `python -c "import torch; torch.cuda.init()"` 看具体错误信息；常见原因是 conda 环境和 pip 装的环境不一致。

---

## 七、驱动与 CUDA 版本对应

| 驱动版本 | 最高支持 CUDA |
|----------|---------------|
| 450.80+ | 11.0 |
| 460+ | 11.2 |
| 470+ | 11.4 |
| 495+ | 11.5 |
| 510+ | 11.6 |
| 515+ | 11.7 |
| 525+ | 12.0 |
| 530+ | 12.1 |
| 535+ | 12.2 |

---

## 八、不想用 GPU？

装 CPU 版 PyTorch（§四.3）即可，训练慢但功能完整。`team_train.py` 会自动检测并标为 `L1`（最低硬件等级）。
