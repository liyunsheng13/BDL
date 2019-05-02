# Bidirectional Learning for Domain Adaptation of Semantic Segmentation
A [pytorch](http://pytorch.org/) implementation of [BDL](https://arxiv.org/pdf/1904.10620.pdf).
If you use this code in your research please consider citing
>@article{li2019bidirectional,
  title={Bidirectional Learning for Domain Adaptation of Semantic Segmentation},
  author={Li, Yunsheng and Yuan, Lu and Vasconcelos, Nuno},
  journal={arXiv preprint arXiv:1904.10620},
  year={2019}
}
### Requirements

- Hardware: PC with NVIDIA Titan GPU.
- Software: *Ubuntu 16.04*, *CUDA 9.2*, *Anaconda2*, *pytorch 0.4.0*
- Python package
  - `conda install pytorch=0.4.0 torchvision cuda91 -y -c pytorch`
  - `pip install tensorboard tensorboardX`

### Datasets
* Download the [GTA5 Dataset](https://download.visinf.tu-darmstadt.de/data/from_games/) as source dataset
* Download the [Cityscapes Dataset](https://www.cityscapes-dataset.com/) as target dataset

### Train adaptive segmenation network in BDL
* Transferred images for CityScapes dataset can be found:
  * [GTA5 as CityScapes](https://drive.google.com/open?id=1OBvYVz2ND4ipdfnkhSaseT8yu2ru5n5l)
* Initial model can be downloaded from [DeepLab-V2](https://drive.google.com/open?id=1TIrTmFKqEyf3pOKniv8-53m3v9SyBK0u)
* Training example (without self-supervised learning):

```
python BDL.py --snapshot-dir ./snapshots/gta2city \
              --init-weights /path/to/inital_weights \
              --num-steps-stop 80000 \
              --model DeepLab
```
* Training example (with self-supervised learning):
  * Download the model [SSL_step1](https://drive.google.com/open?id=1cB5WT_aEm1KYfdOkD81B3C7mRo9ubfcX) or [SSL_step2](https://drive.google.com/open?id=1WtM2dLtRwFvCL_t9Gi6mYhTago_Fk-rA) to generate pseudo labels for CityScapes dataset and then run: 

```
python SSL.py --data-list-target /path/to/dataset/cityscapes_list/train.txt \
              --restore-from /path/to/SSL_step1_or_SSL_step2 \
              --model DeepLab \ 
              --save /path/to/cityscapes/cityscapes_ssl
              --set train
```

With the pseudo labels, the adaptive segmenation model can be trained as:

```
python BDL.py --data-label-folder-target pseudo_label_folder_name \ 
              --snapshot-dir ./snapshots/gta2city_ssl \
              --init-weights /path/to/inital_weights \
              --num-steps-stop 120000 \
              --model DeepLab
```
### Evaluation

```
python evaluation.py --restore-from ./snapshots/gta2city \
                     --save /path/to/cityscapes/results
```
### Acknowledgment
This code is heavily borrowed from [AdaptSegNet](https://github.com/wasidennis/AdaptSegNet)

