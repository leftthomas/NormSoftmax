# SREML
A PyTorch implementation of SREML based on the paper [Squeezed Randomized Ensembles for Metric Learning]().

## Requirements
- [Anaconda](https://www.anaconda.com/download/)
- [PyTorch](https://pytorch.org)
```
conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
```
- capsule-layer
```
pip install git+https://github.com/leftthomas/CapsuleLayer.git@master
```
- pretrainedmodels
```
pip install pretrainedmodels
```

## Datasets
[CARS196](http://ai.stanford.edu/~jkrause/cars/car_dataset.html), [CUB200-2011](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html), 
[Standard Online Products](http://cvgl.stanford.edu/projects/lifted_struct/) and 
[In-shop Clothes](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/InShopRetrieval.html) are used in this repo.

You should download these datasets by yourself, and extract them into `data` directory, make sure the dir names are 
`car`, `cub`, `sop` and `isc`. Then run `data_utils.py` to preprocess them.

## Usage
### Train Model
```
python train.py --data_name cub --crop_type cropped --model_type resnet34 --num_epochs 30
optional arguments:
--data_name                   dataset name [default value is 'car'](choices=['car', 'cub', 'sop', 'isc'])
--crop_type                   crop data or not, it only works for car or cub dataset [default value is 'uncropped'](choices=['uncropped', 'cropped'])
--recalls                     selected recall [default value is '1,2,4,8,10,20,30,40,50,100,1000']
--model_type                  backbone type [default value is 'resnet18'](choices=['resnet18', 'resnet34', 'resnet50', 'resnext50_32x4d', 'se_resnet50', 'se_resnext50_32x4d'])
--batch_size                  train batch size [default value is 12]
--num_epochs                  train epochs number [default value is 20]
--ensemble_size               ensemble model size [default value is 48]
--meta_class_size             meta class size [default value is 12]
--gpu_ids                     selected gpu [default value is '0,1,2']
```

## Benchmarks
Adam optimizer is used with learning rate scheduling. The models are trained with batch size `12`, `20` epochs on three 
NVIDIA Tesla V100 (32G) GPUs.

The images are preprocessed with resize (256, 256), random horizontal flip and normalize. 

For `CARS196` and `CUB200` datasets, ensemble size `48` and meta class size `12` are used. 

For `SOP` dataset, ensemble size `48` and meta class size `256` is used.

For `In-shop` dataset, ensemble size `48` and meta class size `192` is used.

Here is the model parameter details:
<table>
  <thead>
    <tr>
      <th>Backbone</th>
      <th>resnet18</th>
      <th>resnet34</th>
      <th>resnet50</th>
      <th>resnext50_32x4d</th>
      <th>se_resnet50</th>
      <th>se_resnext50_32x4d</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center">CARS196</td>
      <td align="center">529,365,376</td>
      <td align="center">/</td>
      <td align="center">/</td>
      <td align="center">/</td>
      <td align="center">/</td>
      <td align="center">/</td>
    </tr>
    <tr>
      <td align="center">CUB200</td>
      <td align="center">529,365,376</td>
      <td align="center">/</td>
      <td align="center">/</td>
      <td align="center">/</td>
      <td align="center">/</td>
      <td align="center">/</td>
    </tr>
    <tr>
      <td align="center">SOP</td>
      <td align="center">535,373,632</td>
      <td align="center">/</td>
      <td align="center">/</td>
      <td align="center">/</td>
      <td align="center">/</td>
      <td align="center">/</td>
    </tr>
    <tr>
      <td align="center">In-shop</td>
      <td align="center">533,797,696</td>
      <td align="center">/</td>
      <td align="center">/</td>
      <td align="center">/</td>
      <td align="center">/</td>
      <td align="center">/</td>
    </tr>
  </tbody>
</table>

Here is the recall details of `resnet18` backbone:

<table>
  <thead>
    <tr>
      <th>Dataset</th>
      <th>CARS196</th>
      <th>CUB200</th>
      <th>CARS196 (Crop)</th>
      <th>CUB200 (Crop)</th>
      <th>SOP</th>
      <th>In-shop</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center">R@1</td>
      <td align="center">/</td>
      <td align="center">/</td>
      <td align="center">/</td>
      <td align="center">/</td>
      <td align="center">/</td>
      <td align="center">/</td>
    </tr>
    <tr>
      <td align="center">R@2</td>
      <td align="center">/</td>
      <td align="center">/</td>
      <td align="center">/</td>
      <td align="center">/</td>
      <td align="center">/</td>
      <td align="center">/</td>
    </tr>
    <tr>
      <td align="center">R@4</td>
      <td align="center">/</td>
      <td align="center">/</td>
      <td align="center">/</td>
      <td align="center">/</td>
      <td align="center">/</td>
      <td align="center">/</td>
    </tr>
    <tr>
      <td align="center">R@8</td>
      <td align="center">/</td>
      <td align="center">/</td>
      <td align="center">/</td>
      <td align="center">/</td>
      <td align="center">/</td>
      <td align="center">/</td>
    </tr>
    <tr>
      <td align="center">R@10</td>
      <td align="center">/</td>
      <td align="center">/</td>
      <td align="center">/</td>
      <td align="center">/</td>
      <td align="center">/</td>
      <td align="center">/</td>
    </tr>
    <tr>
      <td align="center">R@20</td>
      <td align="center">/</td>
      <td align="center">/</td>
      <td align="center">/</td>
      <td align="center">/</td>
      <td align="center">/</td>
      <td align="center">/</td>
    </tr>
    <tr>
      <td align="center">R@30</td>
      <td align="center">/</td>
      <td align="center">/</td>
      <td align="center">/</td>
      <td align="center">/</td>
      <td align="center">/</td>
      <td align="center">/</td>
    </tr>
    <tr>
      <td align="center">R@40</td>
      <td align="center">/</td>
      <td align="center">/</td>
      <td align="center">/</td>
      <td align="center">/</td>
      <td align="center">/</td>
      <td align="center">/</td>
    </tr>
    <tr>
      <td align="center">R@50</td>
      <td align="center">/</td>
      <td align="center">/</td>
      <td align="center">/</td>
      <td align="center">/</td>
      <td align="center">/</td>
      <td align="center">/</td>
    </tr>
    <tr>
      <td align="center">R@100</td>
      <td align="center">/</td>
      <td align="center">/</td>
      <td align="center">/</td>
      <td align="center">/</td>
      <td align="center">/</td>
      <td align="center">/</td>
    </tr>
    <tr>
      <td align="center">R@1000</td>
      <td align="center">/</td>
      <td align="center">/</td>
      <td align="center">/</td>
      <td align="center">/</td>
      <td align="center">/</td>
      <td align="center">/</td>
    </tr>
  </tbody>
</table>


