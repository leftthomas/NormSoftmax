# REIR
A PyTorch implementation of REIR based on the paper [Randomized Ensembles for Image Retrieval]().

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
      <td align="center">541,381,888</td>
      <td align="center">533,797,696</td>
    </tr>
    <tr>
      <td align="center">ResNet34</td>
      <td align="center">1,011,079,808</td>
      <td align="center">1,011,079,808</td>
      <td align="center">1,023,096,320</td>
      <td align="center">1,015,512,128</td>
    </tr>
    <tr>
      <td align="center">ResNet50</td>
      <td align="center">1,118,974,592</td>
      <td align="center">1,118,974,592</td>
      <td align="center">1,166,970,368</td>
      <td align="center">1,136,677,952</td>
    </tr>
    <tr>
      <td align="center">ResNeXt50</td>
      <td align="center">1,094,093,696</td>
      <td align="center">1,094,093,696</td>
      <td align="center">1,142,089,472</td>
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

![CAR/CUB_Uncropped](results/sota_car_cub.pdf)

- CAR/CUB (Cropped)

![CAR/CUB_Cropped](results/sota_car_cub_crop.pdf)

- SOP

![SOP](results/sota_sop.pdf)

- ISC

![ISC](results/sota_sop.pdf)
