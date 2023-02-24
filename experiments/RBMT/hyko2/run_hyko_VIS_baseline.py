#! /bin/sh

python RBMT_unet_hyko2vis_sem_baseline.py;
sleep 3;
python RBMT_unet_hyko2vis_sem_WD-0_01.py
sleep 3;
python RBMT_unet_hyko2vis_sem_WD-0_05.py
sleep 3;
python RBMT_unet_hyko2vis_sem_WD-0_1.py
sleep 3;
python RBMT_unet_hyko2vis_sem_CW-INS.py 
sleep 3;
python RBMT_unet_hyko2vis_sem_CW-ISNS.py
sleep 3;
python RBMT_unet_hyko2vis_sem_BN.py
sleep 3;

