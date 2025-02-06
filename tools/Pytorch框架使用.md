# 环境安装

## GPU
- 查询本机CUDA版本：
  - 桌面右键 >> NVIDIA控制面板 >> 系统信息 >> 驱动程序版本
  - CMD >> nvidia-smi >> Driver Version
- 查询CUDA版本对应 Toolkit版本 `https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html`
- `>> nvidia-smi` 查询本机GPU**所支持的最高 CUDA 版本**；
- `>> nvcc -V`  查询**已经安装的 CUDA 版本**；
- [CUDA](https://www.nvidia.cn/geforce/drivers/)
- [cuda toolkit URL](https://developer.nvidia.com/cuda-toolkit-archive)
- [torch/torchvision whl URL](https://download.pytorch.org/whl/torch_stable.html)

- 安装流程：
  - 先安装CUDA驱动，安装成功后，即可使用 `nvcc -V` 命令查看版本信息；
  - 然后安装 `torch / torchvision`
  - 安装完测试GPU是否可用 `torch.cuda.is_available()`
  - 
  

## 模型文件读写

### 方法一
```python

```

### 方法二
```python
```

### 方法三：CheckPoint
```python
```