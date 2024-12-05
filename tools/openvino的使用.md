# Python

## 环境搭建

```bash
python -m venv openvino_env
openvino_env/Scripts/activate
python -m pip install --upgrade pip  ??

pip install -q "openvino>=2024.0.0" "nncf>=2.9.0"
pip install -q "torch>=2.1" "torchvision>=0.16" "ultralytics==8.3.0" onnx tqdm opencv-python --extra-index-url https://download.pytorch.org/whl/cpu

python -c "from openvino.runtime import Core"  # 如果没有输出即成功，有报错信息则需要下载官方包进行初始化
```

- 下载官方库：`https://storage.openvinotoolkit.org/repositories/openvino/packages/2024.4/windows/w_openvino_toolkit_windows_2024.4.0.16579.c3152d32c9c_x86_64.zip`
- 解压后，在虚拟环境下执行脚本，`setupvars.bat`



```python
import os
import cv2
import numpy as np
import torch
import openvino as ov

if __name__ == "__main__":
    core = ov.Core()

    model_path = r"D:\share_dir\cell_det\workdir\runs\segment\cell_seg_s\weights\best.onnx"
    model = core.read_model(model=model_path)

    compiled_model = core.compile_model(model=model, device_name="CPU")
    infer_request = compiled_model.create_infer_request()

    input_shape = compiled_model.input(0).shape
    input_data = np.random.rand(*input_shape).astype(np.float32)
    output_layer = compiled_model.output(0)
    result_infer = compiled_model([input_data])[output_layer]

```

# CPP

## openvino runtime
1. 创建 core
2. 创建 compiled_model
3. 创建 infer_request
4. 设置输入图像
5. 推理
6. 解析输出结果

## 图像前处理
### 分类问题常用的图像前处理：
```cpp

    size_t model_input_width = 224;
    size_t model_input_height = 224;
    cv::Mat rgb, blob;
    cv::cvtColor(org_img, rgb, cv::COLOR_BGR2RGB);
    cv::resize(rgb, blob, cv::Size(model_input_width, model_input_height));
    blob.convertTo(blob, CV_32F);
    blob = blob / 255.0;
    cv::subtract(blob, cv::Scalar(0.485, 0.456, 0.406), blob);
    cv::divide(blob, cv::Scalar(0.229,0.224,0.225), blob);  
```


# 部署
- 下载链接：`https://storage.openvinotoolkit.org/repositories/openvino/packages/`
- `w_openvino_toolkit_windows_2023.2.0.13089` 版本发生在有的电脑上不能运行的问题，更换`w_openvino_toolkit_windows_2024.2.0.15519`版本后正常；
- 