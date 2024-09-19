set SEGVOL_CKPT=C:\Users\24305\Documents\v1.pth
set WORK_DIR=.\work_dir
set DATA_DIR=D:\JHU summerproject

python train.py 
--resume %SEGVOL_CKPT% 
--work_dir %WORK_DIR% 
--data_dir %DATA_DIR%
