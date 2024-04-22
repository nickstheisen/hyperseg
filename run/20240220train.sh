#python train.py logging.project_name=hs3bench training.early_stopping=True dataset=hsidrive model=hs3-baseline
#python train.py logging.project_name=hs3bench training.early_stopping=True dataset=hsidrive model=unet
#python train.py logging.project_name=hs3bench training.early_stopping=True dataset=hcv2pca1 model=hs3-baseline model.optimizer_eps=1e-04
#python train.py logging.project_name=hs3bench training.early_stopping=True dataset=hcv2 model=hs3-baseline model.optimizer_eps=1e-04
python train.py logging.project_name=hs3bench training.early_stopping=True training.max_epochs=100 dataset=hcv2_hr model=unet model.optimizer_eps=1e-04
python train.py logging.project_name=hs3bench training.early_stopping=True training.max_epochs=100 dataset=hcv2_hr model=hs3-baseline model.optimizer_eps=1e-04
#python train.py logging.project_name=hs3bench training.early_stopping=True dataset=hyko2 model=unet
#python test.py logging.project_name=hs3bench training.early_stopping=True dataset=hyko2 model=hs3-baseline model.ckpt=/mnt/data/logs/hs3bench/lightning_logs/hyko2-unet-20240227_11-31-32/checkpoints/checkpoint-unet-epoch-209-val-iou-0.557.ckpt
