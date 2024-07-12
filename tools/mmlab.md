# MMLab
- 先获取完整config.py 文件

## MMOCR

### 搭建环境
- 注意不能一股脑安装，需要查看官方的版本依赖关系，指定`MMEngine  MMCV  MMDetection` 版本安装
- 最后 mmocr 采用源码安装方式
- 查看版本依赖：`https://mmocr.readthedocs.io/en/dev-1.x/get_started/install.html`
```bash
conda create -n openmmlab python=3.8 -y
conda activate openmmlab
conda install pytorch torchvision -c pytorch

pip install -U openmim
mim install mmengine
mim install mmcv==2.0.0rc4
mim install mmdet==3.0.0rc5
pip install -v -e .

# 测试安装是否成功
>>> from mmocr.apis import MMOCRInferencer
>>> ocr = MMOCRInferencer(det='DBNet', rec='CRNN')
>>> ocr('demo/demo_text_ocr.jpg', show=True, print_result=True)
```



### 训练
- 数据集：`https://mmocr.readthedocs.io/en/v0.2.0/datasets.html`
- 