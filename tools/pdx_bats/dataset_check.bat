rem E:\le_paddlex\ppx_env\Scripts\python.exe ..\PaddleX\main.py -c PP-HGNetV2-B4.yaml 
rem E:\le_paddlex\PaddleX\paddlex\configs\image_classification\PP-LCNetV2_large.yaml
rem use python -m venv ppx_env   to create python virutal enviroment
E:\le_paddlex\ppx_env\Scripts\python.exe E:\le_paddlex\PaddleX\main.py ^
-c E:\le_paddlex\PaddleX\paddlex\configs\image_classification\PP-HGNetV2-B6.yaml ^
-o Global.mode=check_dataset ^
-o Global.dataset_dir=E:\DataSets\tuffy_crack\pdx_clas  