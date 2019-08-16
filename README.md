# AREIR
A PyTorch implementation of AREIR based on the paper [Attentional Randomized Ensembles for Image Retrieval]().

## Requirements
- [Anaconda](https://www.anaconda.com/download/)
- [PyTorch](https://pytorch.org)
```
conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
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
--model_type                  backbone type [default value is 'resnet18'](choices=['resnet18', 'resnet34', 'resnet50', 'resnext50_32x4d'])
--with_se                     use se block or not [default value is 'yes'](choices=['yes', 'no'])
--batch_size                  train batch size [default value is 8]
--num_epochs                  train epochs number [default value is 20]
--ensemble_size               ensemble model size [default value is 48]
--meta_class_size             meta class size [default value is 12]
--gpu_ids                     selected gpu [default value is '0,1']
```

### Inference Demo
```
python inference.py --retrieval_num 10
optional arguments:
--query_img_name              query image name [default value is 'data/car/uncropped/008055.jpg']
--data_base                   queried database [default value is 'car_uncropped_resnet18_48_12_data_base.pth']
--retrieval_num               retrieval number [default value is 8]
```

## Benchmarks
Adam optimizer is used with learning rate scheduling. The models are trained with batch size `8`, `20` epochs on two 
NVIDIA Tesla V100 (32G) GPUs.

The images are preprocessed with resize (256, 256), random horizontal flip and normalize. 

For `CARS196` and `CUB200` datasets, ensemble size `48` and meta class size `12` are used. 

For `SOP` dataset, ensemble size `48` and meta class size `500` is used.

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
      <th>se-resnet18</th>
      <th>se-resnet34</th>
      <th>se-resnet50</th>
      <th>se-resnext50_32x4d</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center">CARS196</td>
      <td align="center">529,365,376</td>
      <td align="center">1,011,079,808</td>
      <td align="center">1,118,974,592</td>
      <td align="center">1,094,093,696</td>
      <td align="center">529,561,984</td>
      <td align="center">1,011,276,416</td>
      <td align="center">1,122,120,320</td>
      <td align="center">1,097,239,424</td>
    </tr>
    <tr>
      <td align="center">CUB200</td>
      <td align="center">529,365,376</td>
      <td align="center">1,011,079,808</td>
      <td align="center">1,118,974,592</td>
      <td align="center">1,094,093,696</td>
      <td align="center">529,561,984</td>
      <td align="center">1,011,276,416</td>
      <td align="center">1,122,120,320</td>
      <td align="center">1,097,239,424</td>
    </tr>
    <tr>
      <td align="center">SOP</td>
      <td align="center">/</td>
      <td align="center">/</td>
      <td align="center">/</td>
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
      <td align="center">/</td>
      <td align="center">/</td>
    </tr>
  </tbody>
</table>

Here is the results of uncropped `CARS196` dataset:

<table>
  <thead>
    <tr>
      <th>Backbone</th>
      <th>resnet18</th>
      <th>resnet34</th>
      <th>resnet50</th>
      <th>resnext50_32x4d</th>
      <th>se-resnet18</th>
      <th>se-resnet34</th>
      <th>se-resnet50</th>
      <th>se-resnext50_32x4d</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center">R@1</td>
      <td align="center">93.16%</td>
      <td align="center">/</td>
      <td align="center">/</td>
      <td align="center">/</td>
      <td align="center">93.28%</td>
      <td align="center">93.64%</td>
      <td align="center">93.57%</td>
      <td align="center">/</td>
    </tr>
    <tr>
      <td align="center">R@2</td>
      <td align="center">96.56%</td>
      <td align="center">/</td>
      <td align="center">/</td>
      <td align="center">/</td>
      <td align="center">96.32%</td>
      <td align="center">96.40%</td>
      <td align="center">96.14%</td>
      <td align="center">/</td>
    </tr>
    <tr>
      <td align="center">R@4</td>
      <td align="center">98.28%</td>
      <td align="center">/</td>
      <td align="center">/</td>
      <td align="center">/</td>
      <td align="center">98.07%</td>
      <td align="center">98.06%</td>
      <td align="center">97.69%</td>
      <td align="center">/</td>
    </tr>
    <tr>
      <td align="center">R@8</td>
      <td align="center">99.00%</td>
      <td align="center">/</td>
      <td align="center">/</td>
      <td align="center">/</td>
      <td align="center">98.97%</td>
      <td align="center">98.91%</td>
      <td align="center">98.76%</td>
      <td align="center">/</td>
    </tr>
  </tbody>
</table>

Here is the results of cropped `CARS196` dataset:

<table>
  <thead>
    <tr>
      <th>Backbone</th>
      <th>resnet18</th>
      <th>resnet34</th>
      <th>resnet50</th>
      <th>resnext50_32x4d</th>
      <th>se-resnet18</th>
      <th>se-resnet34</th>
      <th>se-resnet50</th>
      <th>se-resnext50_32x4d</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center">R@1</td>
      <td align="center">96.36%</td>
      <td align="center">/</td>
      <td align="center">/</td>
      <td align="center">/</td>
      <td align="center">96.69%</td>
      <td align="center">96.46%</td>
      <td align="center">96.37%</td>
      <td align="center">/</td>
    </tr>
    <tr>
      <td align="center">R@2</td>
      <td align="center">98.13%</td>
      <td align="center">/</td>
      <td align="center">/</td>
      <td align="center">/</td>
      <td align="center">98.24%</td>
      <td align="center">98.01%</td>
      <td align="center">97.97%</td>
      <td align="center">/</td>
    </tr>
    <tr>
      <td align="center">R@4</td>
      <td align="center">98.91%</td>
      <td align="center">/</td>
      <td align="center">/</td>
      <td align="center">/</td>
      <td align="center">98.99%</td>
      <td align="center">98.89%</td>
      <td align="center">98.72%</td>
      <td align="center">/</td>
    </tr>
    <tr>
      <td align="center">R@8</td>
      <td align="center">99.40%</td>
      <td align="center">/</td>
      <td align="center">/</td>
      <td align="center">/</td>
      <td align="center">99.36%</td>
      <td align="center">99.25%</td>
      <td align="center">99.26%</td>
      <td align="center">/</td>
    </tr>
  </tbody>
</table>

Here is the results of uncropped `CUB200` dataset:

<table>
  <thead>
    <tr>
      <th>Backbone</th>
      <th>resnet18</th>
      <th>resnet34</th>
      <th>resnet50</th>
      <th>resnext50_32x4d</th>
      <th>se-resnet18</th>
      <th>se-resnet34</th>
      <th>se-resnet50</th>
      <th>se-resnext50_32x4d</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center">R@1</td>
      <td align="center">79.05%</td>
      <td align="center">/</td>
      <td align="center">/</td>
      <td align="center">/</td>
      <td align="center">79.05%</td>
      <td align="center">71.83%</td>
      <td align="center">71.49%</td>
      <td align="center">/</td>
    </tr>
    <tr>
      <td align="center">R@2</td>
      <td align="center">86.51%</td>
      <td align="center">/</td>
      <td align="center">/</td>
      <td align="center">/</td>
      <td align="center">86.75%</td>
      <td align="center">81.01%</td>
      <td align="center">80.81%</td>
      <td align="center">/</td>
    </tr>
    <tr>
      <td align="center">R@4</td>
      <td align="center">91.37%</td>
      <td align="center">/</td>
      <td align="center">/</td>
      <td align="center">/</td>
      <td align="center">91.63%</td>
      <td align="center">87.32%</td>
      <td align="center">87.14%</td>
      <td align="center">/</td>
    </tr>
    <tr>
      <td align="center">R@8</td>
      <td align="center">94.85%</td>
      <td align="center">/</td>
      <td align="center">/</td>
      <td align="center">/</td>
      <td align="center">95.26%</td>
      <td align="center">91.69%</td>
      <td align="center">91.95%</td>
      <td align="center">/</td>
    </tr>
  </tbody>
</table>

Here is the results of cropped `CUB200` dataset:

<table>
  <thead>
    <tr>
      <th>Backbone</th>
      <th>resnet18</th>
      <th>resnet34</th>
      <th>resnet50</th>
      <th>resnext50_32x4d</th>
      <th>se-resnet18</th>
      <th>se-resnet34</th>
      <th>se-resnet50</th>
      <th>se-resnext50_32x4d</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center">R@1</td>
      <td align="center">87.63%</td>
      <td align="center">/</td>
      <td align="center">/</td>
      <td align="center">/</td>
      <td align="center">87.09%</td>
      <td align="center">85.04%</td>
      <td align="center">84.54%</td>
      <td align="center">/</td>
    </tr>
    <tr>
      <td align="center">R@2</td>
      <td align="center">92.30%</td>
      <td align="center">/</td>
      <td align="center">/</td>
      <td align="center">/</td>
      <td align="center">92.10%</td>
      <td align="center">90.33%</td>
      <td align="center">90.01%</td>
      <td align="center">/</td>
    </tr>
    <tr>
      <td align="center">R@4</td>
      <td align="center">95.53%</td>
      <td align="center">/</td>
      <td align="center">/</td>
      <td align="center">/</td>
      <td align="center">94.89%</td>
      <td align="center">93.79%</td>
      <td align="center">93.38%</td>
      <td align="center">/</td>
    </tr>
    <tr>
      <td align="center">R@8</td>
      <td align="center">97.30%</td>
      <td align="center">/</td>
      <td align="center">/</td>
      <td align="center">/</td>
      <td align="center">97.10%</td>
      <td align="center">96.27%</td>
      <td align="center">95.86%</td>
      <td align="center">/</td>
    </tr>
  </tbody>
</table>

Here is the results of `SOP` dataset:

<table>
  <thead>
    <tr>
      <th>Backbone</th>
      <th>resnet18</th>
      <th>resnet34</th>
      <th>resnet50</th>
      <th>resnext50_32x4d</th>
      <th>se-resnet18</th>
      <th>se-resnet34</th>
      <th>se-resnet50</th>
      <th>se-resnext50_32x4d</th>
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
      <td align="center">/</td>
      <td align="center">/</td>
    </tr>
  </tbody>
</table>

Here is the results of `In-shop` dataset:

<table>
  <thead>
    <tr>
      <th>Backbone</th>
      <th>resnet18</th>
      <th>resnet34</th>
      <th>resnet50</th>
      <th>resnext50_32x4d</th>
      <th>se-resnet18</th>
      <th>se-resnet34</th>
      <th>se-resnet50</th>
      <th>se-resnext50_32x4d</th>
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
      <td align="center">/</td>
      <td align="center">/</td>
    </tr>
  </tbody>
</table>

