# CycleGAN with semantic loss
Download [CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) and replace the files in model/ with ours.
You can use standard folllowing parameters to train it. The initial weights for semantic model for CycleGAN can be downloaded here:
* Download [CycleGAN_sem](https://drive.google.com/open?id=1ysOLKt7vgRjSOCiw1_R2v15R6-qfl3Fr)
```
python train.py --resize_or_crop scale_width_and_crop
                --checkpoints_dir /path/to/checkpoints/ 
                --dataroot datasets/gta2city/ 
                --model cycle_gan 
                --display_freq 3000 
                --save_latest_freq 5000 
                --name gta2city_cyclegan 
                --niter 10 
                --niter_decay 10 
                --loadSize 1024 
                --fineSize 452
                --init_weights /path/to/initial_wegiths 
                --lambda_semantic 1
                --save_epoch_freq 1 
                --gpu 0,1,2,3
                --batchSize 4
                --lambda_identity 0.5
```
