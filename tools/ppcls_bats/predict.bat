set workdir=E:\le_ppcls\crack_cls\PPLCNetV2_large_timmaug_cutmix\best_model
set test_images=E:\DataSets\edge_crack\tmp\2024_9_4

python E:\le_ppcls\PaddleClas\tools\infer.py -c config.yaml ^
 -o Global.device=cpu ^
 -o Global.pretrained_model=%workdir% ^
 -o Global.output_dir=%workdir% ^
 -o Infer.infer_imgs=%test_images%