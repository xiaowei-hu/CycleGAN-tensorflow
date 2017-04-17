<img src='imgs/horse2zebra.gif' align="right" width=384>

<br><br><br>

# CycleGAN

Tensorflow implementation for learning an image-to-image translation **without** input-output pairs,
see [paper](https://arxiv.org/pdf/1703.10593.pdf) , for example:

<img src="https://junyanz.github.io/CycleGAN/images/teaser_high_res.jpg" width="1000px"/>

## Applications
### Monet Paintings to Photos
<img src="imgs/painting2photo.jpg" width="1000px"/>

### Collection Style Transfer
<img src="imgs/photo2painting.jpg" width="1000px"/>

### Object Transfiguration
<img src="imgs/objects.jpg" width="1000px"/>

### Season Transfer
<img src="imgs/season.jpg" width="1000px"/>

### Photo Enhancement: iPhone photo to DSLR photo
<img src="imgs/photo_enhancement.jpg" width="1000px"/>



## Prerequisites
tensorflow r1.0 or higher version

## Getting Started
### Installation
- Install tensorflow and dependencies from https://github.com/tensorflow/tensorflow
- Clone this repo:
```bash
git clone https://github.com/xhujoy/CycleGAN-tensorflow
cd CycleGAN-tensorflow
```

### Train
- Download a dataset (e.g. zebra and horse images from ImageNet):
```bash
bash ./download_dataset.sh horse2zebra
```
- Train a model:
```bash
python main.py --dataset_dir=horse2zebra
```

### Test
- Finally, test the model:
```bash
python main.py --dataset_dir=horse2zebra --phase=test --which_direction=AtoB
```


## Model Zoo
Download the pre-trained models with the following script.

- `orange2apple` (orange -> apple) and `apple2orange`: trained on ImageNet categories `apple` and `orange`.
- `horse2zebra` (horse -> zebra) and `zebra2horse` (zebra -> horse): trained on ImageNet categories `horse` and `zebra`.
- `style_monet` (landscape photo -> Monet painting style),  `style_vangogh` (landscape photo  -> Van Gogh painting style), `style_ukiyoe` (landscape photo  -> Ukiyo-e painting style), `style_cezanne` (landscape photo  -> Cezanne painting style): trained on paintings and Flickr landscape photos.
- `monet2photo` (Monet paintings -> real landscape): trained on paintings and Flickr landscape photographs.
- `cityscapes_photo2label` (street scene -> label) and `cityscapes_label2photo` (label -> street scene): trained on the Cityscapes dataset.
- `map2sat` (map -> aerial photo) and `sat2map` (aerial photo -> map): trained on Google maps.
- `iphone2dslr_flower` (iPhone photos of flower -> DSLR photos of flower): trained on Flickr photos.

## Training and Test Details
To train a model,  
```bash
python main.py --dataset_dir=/path/to/data/ --
```
Models are saved to `./checkpoints/expt_name` (can be changed by passing `checkpoint_dir=your_dir` in train.lua).  
See `opt_train` in `options.lua` for additional training options.

To test the model,
```bash
python main.py --dataset_dir=/path/to/data/ --phase=test --which_direction=AtoB/BtoA
```

## Datasets
Download the datasets using the following script:
```bash
bash ./datasets/download_dataset.sh dataset_name
```
- `facades`: 400 images from the [CMP Facades dataset](http://cmp.felk.cvut.cz/~tylecr1/facade/).
- `cityscapes`: 2975 images from the [Cityscapes training set](https://www.cityscapes-dataset.com/).
- `maps`: 1096 training images scraped from Google Maps.
- `horse2zebra`: 939 horse images and 1177 zebra images downloaded from [ImageNet](http://www.image-net.org/) using keywords `wild horse` and `zebra`
- `apple2orange`: 996 apple images and 1020 orange images downloaded from [ImageNet](http://www.image-net.org/) using keywords `apple` and `navel orange`.
- `summer2winter_yosemite`: 1273 summer Yosemite images and 854 winter Yosemite images were downloaded using Flickr API. See more details in our paper.
- `monet2photo`, `vangogh2photo`, `ukiyoe2photo`, `cezanne2photo`: The art images were downloaded from [Wikiart](https://www.wikiart.org/). The real photos are downloaded from Flickr using combination of tags *landscape* and *landscapephotography*. The training set size of each class is Monet:1074, Cezanne:584, Van Gogh:401, Ukiyo-e:1433, Photographs:6853.
- `iphone2dslr_flower`: both classe of images were downlaoded from Flickr. The training set size of each class is iPhone:1813, DSLR:3316. See more details in our paper.

## Failure cases
<img align="left" style="padding:10px" src="imgs/failure_putin.jpg" width=320>

Our model does not work well when a test image looks unusual compared to training images, as shown in the left figure.  See more typical failure cases [here](https://junyanz.github.io/CycleGAN/images/failures.jpg). On translation tasks that involve color and texture changes, like many of those reported above, the method often succeeds. We have also explored tasks that require geometric changes, with little success. For example, on the task of `dog<->cat` transfiguration, the learned translation degenerates to making minimal changes to the input. We also observe a lingering gap between the results achievable with paired training data and those achieved by our unpaired method. In some cases, this gap may be very hard -- or even impossible,-- to close: for example, our method sometimes permutes the labels for tree and building in the output of the cityscapes photos->labels task.



## Reference
- The torch implementation of CycleGAN, https://github.com/junyanz/CycleGAN
- The tensorflow implementation of pix2pix, https://github.com/yenchenlin/pix2pix-tensorflow
