rem E:\le_paddlex\ppx_env\Scripts\python.exe ..\PaddleX\main.py -c PP-HGNetV2-B4.yaml 
rem E:\le_paddlex\PaddleX\paddlex\configs\image_classification\PP-LCNetV2_large.yaml
E:\le_paddlex\ppx_env\Scripts\python.exe ..\PaddleX\main.py ^
-c E:\le_paddlex\PaddleX\paddlex\configs\image_classification\PP-HGNetV2-B6.yaml ^
-o Global.mode=predict ^
-o Global.dataset_dir=E:\DataSets\tuffy_crack\pdx_clas  ^
-o Global.device=gpu:0 ^
-o Global.output=output_hgnetv2_b6 ^
-o Predict.model_dir=path/to/best_model/inference ^
-o Predict.input=path/to/test_images