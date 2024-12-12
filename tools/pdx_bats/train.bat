rem E:\le_paddlex\ppx_env\Scripts\python.exe ..\PaddleX\main.py -c PP-HGNetV2-B4.yaml 
rem E:\le_paddlex\PaddleX\paddlex\configs\image_classification\PP-LCNetV2_large.yaml
E:\le_paddlex\ppx_env\Scripts\python.exe E:\le_paddlex\PaddleX\main.py ^
-c E:\le_paddlex\PaddleX\paddlex\configs\image_classification\PP-HGNetV2-B6.yaml ^
-o Global.mode=train ^
-o Global.dataset_dir=E:\DataSets\tuffy_crack\pdx_clas  ^
-o Global.device=gpu:0 ^
-o Global.output=output_hgnetv2_b6 ^
-o Train.num_classes=2 ^
-o Train.epochs_iters=500 ^
-o Train.batch_size=64 ^
-o Train.save_interval=50 ^
-o Train.learning_rate=0.05