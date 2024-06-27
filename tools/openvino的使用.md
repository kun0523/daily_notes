# Python

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

