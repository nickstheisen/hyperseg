#! /bin/sh

python RBMT_unet_hsiroad_baseline.py;
sleep 3;
python RBMT_unet_hsiroad_WD-0_01.py;
sleep 3;
python RBMT_unet_hsiroad_WD-0_05.py;
sleep 3;
python RBMT_unet_hsiroad_WD-0_1.py;
sleep 3;
python RBMT_unet_hsiroad_CW-INS.py;
sleep 3;
python RBMT_unet_hsiroad_CW-ISNS.py;
sleep 3;
python RBMT_unet_hsiroad_BN.py;
sleep 3;
