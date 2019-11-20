# REIR
A PyTorch implementation of REIR based on the paper [Randomized Ensembles for Image Retrieval]().

<div align="center">
  <img src="results/architecture.png"/>
</div>

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
python train.py --data_name cub --crop_type cropped --model_type resnet34 --num_epochs 50
optional arguments:
--data_name                   dataset name [default value is 'car'](choices=['car', 'cub', 'sop', 'isc'])
--crop_type                   crop data or not, it only works for car or cub dataset [default value is 'uncropped'](choices=['uncropped', 'cropped'])
--label_type                  assign label with random method or fixed method [default value is 'fixed'](choices=['fixed', 'random'])
--recalls                     selected recall [default value is '1,2,4,8']
--model_type                  backbone type [default value is 'resnet18'](choices=['resnet18', 'resnet34', 'resnet50', 'resnext50_32x4d'])
--share_type                  shared module type [default value is 'layer1'](choices=['none', 'maxpool', 'layer1', 'layer2', 'layer3', 'layer4'])
--with_random                 with branch random weight or not [default value is False]
--load_ids                    load already generated ids or not [default value is False]
--batch_size                  train batch size [default value is 10]
--num_epochs                  train epochs number [default value is 20]
--ensemble_size               ensemble model size [default value is 48]
--meta_class_size             meta class size [default value is 12]
--gpu_ids                     selected gpu [default value is '0,1']
```

### Inference Demo
```
python inference.py --retrieval_num 10 --data_type train
optional arguments:
--query_img_name              query image name [default value is 'data/car/uncropped/008055.jpg']
--data_base                   queried database [default value is 'car_uncropped_fixed_random_layer1_resnet18_48_12_data_base.pth']
--data_type                   retrieval database type [default value is 'test'](choices=['train', 'test'])
--retrieval_num               retrieval number [default value is 8]
```

### Ablation Study
```
python ablation.py --save_results
optional arguments:
--better_data_base            better database [default value is 'car_uncropped_fixed_unrandom_layer1_resnet18_48_12_data_base.pth']
--worse_data_base             worse database [default value is 'car_uncropped_random_unrandom_layer1_resnet18_48_12_data_base.pth']
--data_type                   retrieval database type [default value is 'test'](choices=['train', 'test'])
--retrieval_num               retrieval number [default value is 8]
--save_results                with save results or not [default value is False]
```

## Benchmarks
Adam optimizer is used with learning rate scheduling. The models are trained with batch size `10` on two 
NVIDIA Tesla V100 (32G) GPUs.

The images are preprocessed with resize (256, 256), random horizontal flip and normalize. 

For `CARS196` and `CUB200` datasets, ensemble size `48`, meta class size `12` and `20` epochs are used. 

For `SOP` dataset, ensemble size `48`, meta class size `512` and `40` epochs are used.

For `In-shop` dataset, ensemble size `48`, meta class size `192` and `40` epochs are used.

### Model Parameter
<table>
  <thead>
    <tr>
      <th>Backbone</th>
      <th>CARS196</th>
      <th>CUB200</th>
      <th>SOP</th>
      <th>In-shop</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center">ResNet18</td>
      <td align="center">529,365,376</td>
      <td align="center">529,365,376</td>
      <td align="center">541,677,376</td>
      <td align="center">533,797,696</td>
    </tr>
    <tr>
      <td align="center">ResNet34</td>
      <td align="center">1,011,079,808</td>
      <td align="center">1,011,079,808</td>
      <td align="center">1,023,391,808</td>
      <td align="center">1,015,512,128</td>
    </tr>
    <tr>
      <td align="center">ResNet50</td>
      <td align="center">1,118,974,592</td>
      <td align="center">1,118,974,592</td>
      <td align="center">1,168,150,592</td>
      <td align="center">1,136,677,952</td>
    </tr>
    <tr>
      <td align="center">ResNeXt50</td>
      <td align="center">1,094,093,696</td>
      <td align="center">1,094,093,696</td>
      <td align="center">1,143,269,696</td>
      <td align="center">1,111,797,056</td>
    </tr>
  </tbody>
</table>

### CARS196 (Uncropped)
<table>
  <thead>
    <tr>
      <th>Backbone</th>
      <th>R@1</th>
      <th>R@2</th>
      <th>R@4</th>
      <th>R@8</th>
      <th>Download Link</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center">ResNet18</td>
      <td align="center">93.4%</td>
      <td align="center">96.6%</td>
      <td align="center">98.1%</td>
      <td align="center">99.0%</td>
      <td align="center"><a href="https://pan.baidu.com/s/1lkek0pPAWGNNZiOAejFCxw">model</a>&nbsp;|&nbsp;sp3q</td>
    </tr>
    <tr>
      <td align="center">ResNet34</td>
      <td align="center">93.5%</td>
      <td align="center">96.2%</td>
      <td align="center">97.7%</td>
      <td align="center">98.8%</td>
      <td align="center"><a href="https://pan.baidu.com/s/1U7rbDRT9XEXBY3VU5goLCA">model</a>&nbsp;|&nbsp;g8k9</td>
    </tr>
    <tr>
      <td align="center">ResNet50</td>
      <td align="center">93.3%</td>
      <td align="center">96.2%</td>
      <td align="center">97.7%</td>
      <td align="center">98.5%</td>
      <td align="center"><a href="https://pan.baidu.com/s/1m91YFmycmD4xwGCDJVJFHQ">model</a>&nbsp;|&nbsp;s4gj</td>
    </tr>
    <tr>
      <td align="center">ResNeXt50</td>
      <td align="center">93.9%</td>
      <td align="center">96.5%</td>
      <td align="center">97.8%</td>
      <td align="center">98.7%</td>
      <td align="center"><a href="https://pan.baidu.com/s/1NVAcxCxIuXBlxW13hf82TQ">model</a>&nbsp;|&nbsp;dcrm</td>
    </tr>
  </tbody>
</table>

### CARS196 (Cropped)
<table>
  <thead>
    <tr>
      <th>Backbone</th>
      <th>R@1</th>
      <th>R@2</th>
      <th>R@4</th>
      <th>R@8</th>
      <th>Download Link</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center">ResNet18</td>
      <td align="center">96.7%</td>
      <td align="center">98.3%</td>
      <td align="center">99.0%</td>
      <td align="center">99.5%</td>
      <td align="center"><a href="https://pan.baidu.com/s/1U3KNMoS0zBErDLV8cYjpYg">model</a>&nbsp;|&nbsp;ttgs</td>
    </tr>
    <tr>
      <td align="center">ResNet34</td>
      <td align="center">96.7%</td>
      <td align="center">98.3%</td>
      <td align="center">99.0%</td>
      <td align="center">99.4%</td>
      <td align="center"><a href="https://pan.baidu.com/s/180KNBTZ_kX2trgShnok_IA">model</a>&nbsp;|&nbsp;htar</td>
    </tr>
    <tr>
      <td align="center">ResNet50</td>
      <td align="center">96.6%</td>
      <td align="center">98.1%</td>
      <td align="center">98.7%</td>
      <td align="center">99.2%</td>
      <td align="center"><a href="https://pan.baidu.com/s/1V8hJylBM0Q2iHSIcQrYaCA">model</a>&nbsp;|&nbsp;kz98</td>
    </tr>
    <tr>
      <td align="center">ResNeXt50</td>
      <td align="center">96.8%</td>
      <td align="center">98.2%</td>
      <td align="center">98.9%</td>
      <td align="center">99.3%</td>
      <td align="center"><a href="https://pan.baidu.com/s/1xusMA7oWp1mEIl3IyYm3aQ">model</a>&nbsp;|&nbsp;9jxx</td>
    </tr>
  </tbody>
</table>

### CUB200 (Uncropped)
<table>
  <thead>
    <tr>
      <th>Backbone</th>
      <th>R@1</th>
      <th>R@2</th>
      <th>R@4</th>
      <th>R@8</th>
      <th>Download Link</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center">ResNet18</td>
      <td align="center">82.0%</td>
      <td align="center">88.9%</td>
      <td align="center">92.6%</td>
      <td align="center">95.6%</td>
      <td align="center"><a href="https://pan.baidu.com/s/1aANPHE8zw3t_5ZHpxMtBTg">model</a>&nbsp;|&nbsp;igua</td>
    </tr>
    <tr>
      <td align="center">ResNet34</td>
      <td align="center">77.5%</td>
      <td align="center">85.0%</td>
      <td align="center">90.4%</td>
      <td align="center">94.3%</td>
      <td align="center"><a href="https://pan.baidu.com/s/19z5kmrIbNb8WGdDIcOmd5g">model</a>&nbsp;|&nbsp;y71x</td>
    </tr>
    <tr>
      <td align="center">ResNet50</td>
      <td align="center">77.8%</td>
      <td align="center">84.9%</td>
      <td align="center">89.9%</td>
      <td align="center">93.7%</td>
      <td align="center"><a href="https://pan.baidu.com/s/1x5ckVuS9pm7hMrynsmaS6w">model</a>&nbsp;|&nbsp;pa8c</td>
    </tr>
    <tr>
      <td align="center">ResNeXt50</td>
      <td align="center">80.1%</td>
      <td align="center">86.8%</td>
      <td align="center">91.5%</td>
      <td align="center">94.8%</td>
      <td align="center"><a href="https://pan.baidu.com/s/19qkoDtZwCdQpN-bJ2FiP9g">model</a>&nbsp;|&nbsp;u37j</td>
    </tr>
  </tbody>
</table>

### CUB200 (Cropped)
<table>
  <thead>
    <tr>
      <th>Backbone</th>
      <th>R@1</th>
      <th>R@2</th>
      <th>R@4</th>
      <th>R@8</th>
      <th>Download Link</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center">ResNet18</td>
      <td align="center">89.0%</td>
      <td align="center">93.1%</td>
      <td align="center">95.9%</td>
      <td align="center">97.5%</td>
      <td align="center"><a href="https://pan.baidu.com/s/10kONUyM4zosjZhEXcix_Qg">model</a>&nbsp;|&nbsp;vn7c</td>
    </tr>
    <tr>
      <td align="center">ResNet34</td>
      <td align="center">86.0%</td>
      <td align="center">91.2%</td>
      <td align="center">94.7%</td>
      <td align="center">96.8%</td>
      <td align="center"><a href="https://pan.baidu.com/s/1eY5_ISeaZyjTKm6r-9yOeA">model</a>&nbsp;|&nbsp;w2m4</td>
    </tr>
    <tr>
      <td align="center">ResNet50</td>
      <td align="center">85.9%</td>
      <td align="center">91.4%</td>
      <td align="center">94.5%</td>
      <td align="center">96.4%</td>
      <td align="center"><a href="https://pan.baidu.com/s/1srBQqU_vYzoTr4Mx7UV6Nw">model</a>&nbsp;|&nbsp;vqcg</td>
    </tr>
    <tr>
      <td align="center">ResNeXt50</td>
      <td align="center">86.3%</td>
      <td align="center">91.3%</td>
      <td align="center">94.5%</td>
      <td align="center">96.6%</td>
      <td align="center"><a href="https://pan.baidu.com/s/14g64iGZCR4Txox2-40SAFQ">model</a>&nbsp;|&nbsp;tkwc</td>
    </tr>
  </tbody>
</table>

### SOP
<table>
  <thead>
    <tr>
      <th>Backbone</th>
      <th>R@1</th>
      <th>R@10</th>
      <th>R@100</th>
      <th>R@1000</th>
      <th>Download Link</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center">ResNet18</td>
      <td align="center">94.47%</td>
      <td align="center">94.23%</td>
      <td align="center">94.32%</td>
      <td align="center">95.01%</td>
      <td align="center"><a href="https://pan.baidu.com/s/1k6wIhyJfRLtr9WY_SoZNIw">model</a>&nbsp;|&nbsp;vqeq</td>
    </tr>
    <tr>
      <td align="center">ResNet34</td>
      <td align="center">97.21%</td>
      <td align="center">96.75%</td>
      <td align="center">96.94%</td>
      <td align="center">97.26%</td>
      <td align="center"><a href="https://pan.baidu.com/s/1k6wIhyJfRLtr9WY_SoZNIw">model</a>&nbsp;|&nbsp;vqeq</td>
    </tr>
    <tr>
      <td align="center">ResNet50</td>
      <td align="center">98.60%</td>
      <td align="center">98.20%</td>
      <td align="center">98.30%</td>
      <td align="center">98.20%</td>
      <td align="center"><a href="https://pan.baidu.com/s/1k6wIhyJfRLtr9WY_SoZNIw">model</a>&nbsp;|&nbsp;vqeq</td>
    </tr>
    <tr>
      <td align="center">ResNeXt50</td>
      <td align="center">99.37%</td>
      <td align="center">98.94%</td>
      <td align="center">98.97%</td>
      <td align="center">98.94%</td>
      <td align="center"><a href="https://pan.baidu.com/s/1k6wIhyJfRLtr9WY_SoZNIw">model</a>&nbsp;|&nbsp;vqeq</td>
    </tr>
  </tbody>
</table>

### In-shop
<table>
  <thead>
    <tr>
      <th>Backbone</th>
      <th>R@1</th>
      <th>R@10</th>
      <th>R@20</th>
      <th>R@30</th>
      <th>R@40</th>
      <th>R@50</th>
      <th>Download Link</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center">ResNet18</td>
      <td align="center">94.47%</td>
      <td align="center">94.23%</td>
      <td align="center">94.32%</td>
      <td align="center">95.01%</td>
      <td align="center">94.32%</td>
      <td align="center">95.01%</td>
      <td align="center"><a href="https://pan.baidu.com/s/1k6wIhyJfRLtr9WY_SoZNIw">model</a>&nbsp;|&nbsp;vqeq</td>
    </tr>
    <tr>
      <td align="center">ResNet34</td>
      <td align="center">97.21%</td>
      <td align="center">96.75%</td>
      <td align="center">96.94%</td>
      <td align="center">97.26%</td>
      <td align="center">94.32%</td>
      <td align="center">95.01%</td>
      <td align="center"><a href="https://pan.baidu.com/s/1k6wIhyJfRLtr9WY_SoZNIw">model</a>&nbsp;|&nbsp;vqeq</td>
    </tr>
    <tr>
      <td align="center">ResNet50</td>
      <td align="center">98.60%</td>
      <td align="center">98.20%</td>
      <td align="center">98.30%</td>
      <td align="center">98.20%</td>
      <td align="center">94.32%</td>
      <td align="center">95.01%</td>
      <td align="center"><a href="https://pan.baidu.com/s/1k6wIhyJfRLtr9WY_SoZNIw">model</a>&nbsp;|&nbsp;vqeq</td>
    </tr>
    <tr>
      <td align="center">ResNeXt50</td>
      <td align="center">99.37%</td>
      <td align="center">98.94%</td>
      <td align="center">98.97%</td>
      <td align="center">98.94%</td>
      <td align="center">94.32%</td>
      <td align="center">95.01%</td>
      <td align="center"><a href="https://pan.baidu.com/s/1k6wIhyJfRLtr9WY_SoZNIw">model</a>&nbsp;|&nbsp;vqeq</td>
    </tr>
  </tbody>
</table>

## Results

- CAR/CUB (Uncropped)

![CAR/CUB_Uncropped](results/sota_car_cub.png)

- CAR/CUB (Cropped)

![CAR/CUB_Cropped](results/sota_car_cub_crop.png)

- SOP

![SOP](results/sota_sop.png)

- ISC

![ISC](results/sota_sop.png)
