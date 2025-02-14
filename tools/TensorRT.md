# TensorRT

## TensorRT CUDA cuDNN 本地版本查询
- TensorRT
  - 查看安装包
  - 使用 `trtexec --version`
- CUDA
  - `nvcc --version`
- cuDNN
  - `cudnn_version.h` 在头文件中
    - `CUDNN_MAJOR 8`
    - `CUDNN_MINOR 9`
    - `CUDNN_PATCHLEVEL 7`

## 三者匹配关系查询
- CUDA 11.8、cuDNN 8.9.0、TensorRT 8.6.1：适用于30系卡  自测试OK
- https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/support-matrix.html
- 
## 安装

1. 安装 CUDA： 下载并运行 NVIDIA 提供的 CUDA 安装程序，按照提示进行安装.
2. 安装 cuDNN： 下载 cuDNN 压缩包，解压后将文件复制到 CUDA 安装目录下的对应位置。例如，将 include 目录中的文件复制到 C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\include，将 lib64 目录中的文件复制到 C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\lib\x64.
3. 安装 TensorRT： 下载 TensorRT 压缩包，解压后按照 README 文件中的指示进行安装。通常，这涉及到设置环境变量和将必要的库文件复制到正确的位置.

### CUDA 安装
- 作用：是NVIDIA的并行计算平台和编程模型，用于在NVIDIA GPU上运行代码
- 下载：https://developer.nvidia.com/cuda-downloads
- 配置：将bin目录配置到环境变量中
- 测试：`nvcc --version`

### cuDNN 安装
- 作用：cuDNN是一个深度神经网络库，需要与CUDA配合使用
- 下载：https://developer.nvidia.com/rdp/cudnn-archive
- 配置：将 `include` `lib\x64` 分别复制到 CUDA 目录下相应的文件夹下

### TensorRT 安装
- 作用：是一个用于优化深度学习模型推理的库
- 下载：https://developer.nvidia.com/tensorrt
- 配置：将bin目录配置到环境变量中
- 测试：`trtexec --version`

### 测试代码

#### Python
- 在TensorRT下载包中，有Python的库安装包 .whl，依次安装，主要python版本
    - `tensorrt-xxx`
    - `tensorrt_dispatch-xxx`
    - `tensorrt_lean-xxx`
    - `uff`
    - `graphsurgeon`
    - `onnx_graphsurgeon`
```python
import tensorrt as trt

print(trt.__version__)
```


#### CPP
- 注意配置 `CUDA` `TensorRT` 头文件和库文件

```cpp
#include <iostream>
#include <opencv2/opencv.hpp>
#include <NvInfer.h>
#include <NvOnnxParser.h>

using namespace nvinfer1;
using namespace nvonnxparser;  // 用于解析onnx
using namespace cv;

// 必须要定义， 创建 IBuilder  IRuntime 时要用
class Logger : public ILogger {
    void log(Severity severity, const char* msg) noexcept {
        if (severity != Severity::kINFO)  // 出错时打印
            std::cout << msg << std::endl;
    }
}gLogger;


int main()
{
    IBuilder* builder = createInferBuilder(gLogger);
    IRuntime* runtime = createInferRuntime(gLogger);    
    builder->getLogger()->log(nvinfer1::ILogger::Severity::kWARNING, "Create Builder...");
    runtime->getLogger()->log(nvinfer1::ILogger::Severity::kWARNING, "Create Runtime...");
    std::cout << "Hello TensorRT!\n";
}
```

## SDK 常用API

- `#include "NvInfer.h"`  推理要用的头文件
- `ILogger`
- `IBuild`  将onnx转为engine形式
- `IRuntime`
- `ICudaEngine`
- `IExecutionContext`  推理上下文
- `cudaMalloc` 显存分配
- `cudaMemcpyAsync`  
  - h2d  host to device 内存 --> GPU
  - d2h  device to host GPU  --> 内存
  - d2d  device to device GPU  --> GPU
  - h2h  内存 --> 内存

- 工作流：
  - 准备阶段：
    - 二进制模式读取 engine 模型文件
    - 创建 `IRuntime` 实例 - `createInferRuntime`
    - 反序列化创建 `ICudaEngine` - `deserializeCudaEngine`  将二进制模型转为 `ICudaEngine`
    - 创建可执行的 `IExecutionContext` - `createExecutionContext`
    - 获取输入与输出层格式与大小，初始化cuda内存分配
  - 推理阶段：
    - 图像预处理-格式化输入图像数据
    - enqueue（V2:image  V3:stream） 推理
    - 预测结果后处理与显示
  - 收尾阶段：
    - 结束释放内存

## `trtexec` 工具使用
- `--onnx=<file>`  指定onnx模型文件
- `--fp16`  
- `saveEngine=<file>`  保持序列号engine模型文件
- `--verbose`  显示详细信息  默认不显示

### onnx 转 engine
- `trtexec --onnx=path/to/model.onnx --saveEngine=path/to/model_fp32.onnx` 
  - `fp32` 精度的文件大小比原onnx文件还大一些
- `trtexec --onnx=path/to/model.onnx --saveEngine=path/to/model_fp16.onnx --fp16`
  - `fp16` 精度的文件大小比原onnx文件小差不多一半
- TODO: 转int8 ... ... 


## 分类任务实例代码
```cpp
#include <iostream>
#include <fstream>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <NvInfer.h>

using namespace nvinfer1;
using namespace cv;

// 必须要定义， 创建 IBuilder  IRuntime 时要用
class Logger : public ILogger {
    void log(Severity severity, const char* msg) noexcept {
        if (severity != Severity::kINFO)  // 出错时打印
            std::cout << msg << std::endl;
    }
}gLogger;

// 解析类别标签
std::string labels_txt_file = R"(E:\le_trt\models\imagenet_classes.txt)";
std::vector<std::string> readClassName() {
    std::vector<std::string> classNames;

    std::ifstream fp(labels_txt_file);
    if (!fp.is_open()) {
        printf("Could not open file...\n");
        exit(-1);
    }
    std::string name;
    while (!fp.eof()) {
        std::getline(fp, name);
        if (name.length())
            classNames.push_back(name);
    }
    fp.close();
    return classNames;
}

int main()
{
    std::vector<std::string> labels = readClassName();
    std::string enginepath = R"(E:\le_trt\models\resnet18.engine)";
    std::ifstream file(enginepath, std::ios::binary);
    char* trtModelStream = nullptr;
    int size = 0;
    if (file.good()) {
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);
        trtModelStream = new char[size];
        assert(trtModelStream);
        file.read(trtModelStream, size);
        file.close();
    }

    auto runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    auto engine = runtime->deserializeCudaEngine(trtModelStream, size);
    assert(engine != nullptr);
    auto context = engine->createExecutionContext();
    assert(context != nullptr);
    delete[] trtModelStream;

    void* buffers[2] = { NULL, NULL };  // 用于保存在GPU中 输入/输出变量的地址
    std::vector<float> prob;
    cudaStream_t stream;

    int input_index = engine->getBindingIndex("input");
    int output_index = engine->getBindingIndex("output");

    // 获取输入维度信息 NCHW
    int input_h = engine->getBindingDimensions(input_index).d[2];
    int input_w = engine->getBindingDimensions(input_index).d[3];
    std::cout << "inputH: " << input_h << " inputW:" << input_w << std::endl;

    // 获取输出维度信息 
    int output_h = engine->getBindingDimensions(output_index).d[0];
    int output_w = engine->getBindingDimensions(output_index).d[1];
    std::cout << "output data format: " << output_h << "x" << output_w << std::endl;

    // 创建GPU显存输入 输出缓冲区
    std::cout << " input/output : " << engine->getNbBindings() << std::endl; // get the number of binding indices
    cudaMalloc(&buffers[input_index], input_h * input_w * 3 * sizeof(float));
    cudaMalloc(&buffers[output_index], output_h * output_w * sizeof(float));

    // 创建临时缓存输出
    prob.resize(output_h * output_w);

    // 创建cuda流
    cudaStreamCreate(&stream);

    // 第一次推理12ms，后续的推理3ms左右
    for (int i = 0; i < 10; i++) {
        auto start = std::chrono::high_resolution_clock::now();

        cv::Mat image = cv::imread(R"(E:\le_trt\models\dog.jpg)");
        cv::Mat rgb, blob;
        cv::cvtColor(image, rgb, cv::COLOR_BGR2RGB);
        cv::resize(rgb, blob, cv::Size(224, 224));
        blob.convertTo(blob, CV_32F);
        blob = blob / 255.0;
        cv::subtract(blob, cv::Scalar(0.485, 0.456, 0.406), blob);
        cv::divide(blob, cv::Scalar(0.229, 0.224, 0.225), blob);

        // HWC -> CHW
        cv::Mat tensor = cv::dnn::blobFromImage(blob);

        // 内存到GPU显存
        cudaMemcpyAsync(buffers[0], tensor.ptr<float>(), input_h * input_w * 3 * sizeof(float), cudaMemcpyHostToDevice, stream);

        // 推理
        context->enqueueV2(buffers, stream, nullptr);

        // GPU显存到内存
        cudaMemcpyAsync(prob.data(), buffers[1], output_h * output_w * sizeof(float), cudaMemcpyDeviceToHost, stream);

        // 后处理
        cv::Mat probmat(output_h, output_w, CV_32F, (float*)prob.data());
        cv::Point maxL, minL;
        double maxv, minv;
        cv::minMaxLoc(probmat, &minv, &maxv, &minL, &maxL);
        int max_index = maxL.x;
        std::cout << "label id: " << max_index << " class: " << labels[max_index] << std::endl;

        auto stop = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double, std::milli> spend = stop - start;
        std::cout << "Iter: " << i << " Inference cost: " << spend.count() << std::endl;
    }

    // 同步结束 释放资源
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);

    if (!context) {
        context->destroy();
    }
    if (!engine) {
        engine->destroy();
    }
    if (!runtime) {
        runtime->destroy();
    }
    if (!buffers[0]) {
        delete[] buffers;
    }

    std::cout << "Job Done" << std::endl;

    return 0;
}

```


## 检测任务实例代码

```cpp
// ConsoleApplication1.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include <iostream>
#include <fstream>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <NvInfer.h>
#include <NvOnnxParser.h>

namespace det_task {

    using namespace nvinfer1;
    using namespace nvonnxparser;  // 用于解析onnx
    using namespace cv;

    // 必须要定义， 创建 IBuilder  IRuntime 时要用
    class Logger : public ILogger {
        void log(Severity severity, const char* msg) noexcept {
            if (severity != Severity::kINFO)  // 出错时打印
                std::cout << msg << std::endl;
        }
    }gLogger;

    // 解析类别标签
    std::string labels_txt_file = R"(E:\le_trt\models\coco_classes.txt)";
    std::vector<std::string> readClassName() {
        std::vector<std::string> classNames;

        std::ifstream fp(labels_txt_file);
        if (!fp.is_open()) {
            printf("Could not open file...\n");
            exit(-1);
        }
        std::string name;
        while (!fp.eof()) {
            std::getline(fp, name);
            if (name.length())
                classNames.push_back(name);
        }
        fp.close();
        return classNames;
    }

}
int main()
{
    std::vector<std::string> labels = det_task::readClassName();
    std::string enginepath = R"(E:\le_trt\models\yolov5s.engine)";
    std::ifstream file(enginepath, std::ios::binary);
    char* trtModelStream = nullptr;
    int size = 0;
    if (file.good()) {
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);
        trtModelStream = new char[size];
        assert(trtModelStream);
        file.read(trtModelStream, size);
        file.close();
    }

    auto runtime = det_task::createInferRuntime(det_task::gLogger);
    assert(runtime != nullptr);
    auto engine = runtime->deserializeCudaEngine(trtModelStream, size);
    assert(engine != nullptr);
    auto context = engine->createExecutionContext();
    assert(context != nullptr);
    delete[] trtModelStream;

    void* buffers[2] = { NULL, NULL };  
    std::vector<float> prob;
    cudaStream_t stream;

    int input_index = engine->getBindingIndex("images");  // 0
    int output_index = engine->getBindingIndex("output0");  // 1
    std::cout << "input_index: " << input_index << " output_index: " << output_index << std::endl;

    // 获取输入维度信息 NCHW 1x3x640x640
    int input_h = engine->getBindingDimensions(input_index).d[2];
    int input_w = engine->getBindingDimensions(input_index).d[3];
    std::cout << "inputH: " << input_h << " inputW:" << input_w << std::endl;

    // 获取输出维度信息 84x8400
    int output_h = engine->getBindingDimensions(output_index).d[1];
    int output_w = engine->getBindingDimensions(output_index).d[2];
    std::cout << "output data format: " << output_h << "x" << output_w << std::endl;

    // 创建GPU显存输入 输出缓冲区
    std::cout << "input/output : " << engine->getNbBindings() << std::endl; // get the number of binding indices
    cudaMalloc(&buffers[input_index], input_h * input_w * 3 * sizeof(float));
    cudaMalloc(&buffers[output_index], output_h * output_w * sizeof(float));

    // 创建零食缓存输出
    prob.resize(output_h * output_w);

    // 创建cuda流
    cudaStreamCreate(&stream);

    float score_threshold = 0.5;
    // 第一次推理12ms，后续的推理3ms左右
    for (int i = 0; i < 10; i++) {
        auto start = std::chrono::high_resolution_clock::now();

        cv::Mat image = cv::imread(R"(E:\le_trt\models\dog.jpg)");
        //cv::Mat rgb, blob;
        //cv::cvtColor(image, rgb, cv::COLOR_BGR2RGB);
        //cv::resize(rgb, blob, cv::Size(input_w, input_h));
        //blob.convertTo(blob, CV_32F);
        //blob = blob / 255.0;
        //cv::subtract(blob, cv::Scalar(0.485, 0.456, 0.406), blob);
        //cv::divide(blob, cv::Scalar(0.229, 0.224, 0.225), blob);
        int w = image.cols;
        int h = image.rows;
        int _max = std::max(h, w);
        cv::Mat tmp = cv::Mat::zeros(cv::Size(_max, _max), CV_8UC3);
        cv::Rect roi(0, 0, w, h);
        image.copyTo(tmp(roi));
        cv::Mat tensor = cv::dnn::blobFromImage(tmp, 1.0f / 255.0f, cv::Size(input_w, input_h), cv::Scalar(), true);

        float x_factor = image.cols / input_w;
        float y_factor = image.rows / input_h;

        // 内存到GPU显存
        cudaMemcpyAsync(buffers[0], tensor.ptr<float>(), input_h * input_w * 3 * sizeof(float), cudaMemcpyHostToDevice, stream);

        // 推理
        context->enqueueV2(buffers, stream, nullptr);

        // GPU显存到内存
        cudaMemcpyAsync(prob.data(), buffers[1], output_h * output_w * sizeof(float), cudaMemcpyDeviceToHost, stream);

        // 后处理
        std::vector<cv::Rect> boxes;
        std::vector<int> classIds;
        std::vector<float> confidences;
        cv::Mat probmat(output_h, output_w, CV_32F, (float*)prob.data());
        cv::Mat res = probmat.t();
        std::cout << res.size << std::endl;
        for (int i{}; i < 5; i++) {
            //std::cout << res.at<float>(i, 0) << " " << res.at<float>(i, 1) << " " << res.at<float>(i, 2) << " " << res.at<float>(i, 3) << " " << res.at<float>(i, 4) << " " << res.at<float>(i, 5) << std::endl;
            cv::Mat classes_scores = res.row(i).colRange(4, output_h);
            cv::Point classIdPoint;
            double score;
            cv::minMaxLoc(classes_scores, 0, &score, 0, &classIdPoint);
            //std::cout << i << "   " << score << "    " << classIdPoint.x << std::endl;

            if (score > score_threshold) {
                float cx = res.at<float>(i, 0);
                float cy = res.at<float>(i, 1);
                float ow = res.at<float>(i, 2);
                float oh = res.at<float>(i, 3);

                int x = static_cast<int>((cx - 0.5 * ow) * x_factor);
                int y = static_cast<int>((cy - 0.5 * oh) * y_factor);
                int width = static_cast<int>(ow * x_factor);
                int height = static_cast<int>(oh * y_factor);
                cv::Rect box;
                box.x = x;
                box.y = y;
                box.width = width;
                box.height = height;
            }

        }
        cv::Point maxL, minL;
        double maxv, minv;
        cv::minMaxLoc(probmat, &minv, &maxv, &minL, &maxL);
        int max_index = maxL.x;
        std::cout << "label id: " << max_index << " class: " << labels[max_index] << std::endl;

        auto stop = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double, std::milli> spend = stop - start;
        std::cout << "Iter: " << i << " Inference cost: " << spend.count() << std::endl;
    }

    // 同步结束 释放资源
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);

    if (!context) {
        context->destroy();
    }
    if (!engine) {
        engine->destroy();
    }
    if (!runtime) {
        runtime->destroy();
    }
    if (!buffers[0]) {
        delete[] buffers;
    }

    std::cout << "Job Done" << std::endl;

    return 0;


}
```


## 分割任务实例代码