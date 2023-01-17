#! /bin/sh

sleep 3;
python RBMT_unet_hyko2nir_sem_baseline.py;
sleep 3;
python RBMT_unet_hyko2nir_sem_wd0-01.py;
sleep 3;
python RBMT_unet_hyko2nir_sem_wd0-1.py;
sleep 3;
python RBMT_unet_hyko2nir_sem_wd0-5.py;
sleep 3;
python RBMT_unet_hyko2vis_sem_baseline.py;
sleep 3;
python RBMT_unet_hyko2vis_sem_wd0-01.py;
sleep 3;
python RBMT_unet_hyko2vis_sem_wd0-1.py;
sleep 3;
python RBMT_unet_hyko2vis_sem_wd0-5.py;
sleep 3;

