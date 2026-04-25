@echo off
echo Resuming training from Epoch 3 checkpoint...
python main.py --mode train --epochs 10 --batch_size 2 --lr 1e-4 --optimizer adamw --scheduler plateau --augmentation --pretrained --freeze_backbone --freeze_layers 2 --early_stopping --early_stopping_patience 3 --grad_accum_steps 4 --resume output/checkpoints/checkpoint_epoch_3.pth
pause
