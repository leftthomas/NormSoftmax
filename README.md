# CLF
A PyTorch implementation of CLF based on the paper [Combination of Multiple Local Features for Image Retrieval]().

![Network Architecture](results/structure.png)

## Requirements
- [Anaconda](https://www.anaconda.com/download/)
- [PyTorch](https://pytorch.org)
```
conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
```
- thop
```
pip install thop
```

## Datasets
[CARS196](http://ai.stanford.edu/~jkrause/cars/car_dataset.html), [CUB200-2011](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html), 
[Standard Online Products](http://cvgl.stanford.edu/projects/lifted_struct/) and 
[In-shop Clothes](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/InShopRetrieval.html) are used in this repo.

You should download these datasets by yourself, and extract them into `${data_path}` directory, make sure the dir names are 
`car`, `cub`, `sop` and `isc`. Then run `data_utils.py` to preprocess them.

## Usage
### Train Model
```
python train.py --backbone_type resnet50 --feature_dim 1024
optional arguments:
--data_path                   datasets path [default value is '/home/data']
--data_name                   dataset name [default value is 'car'](choices=['car', 'cub', 'sop', 'isc'])
--crop_type                   crop data or not, it only works for car or cub dataset [default value is 'uncropped'](choices=['uncropped', 'cropped'])
--backbone_type               backbone network type [default value is 'resnet18'](choices=['resnet18', 'resnet50'])
--feature_dim                 feature dim [default value is 512]
--recalls                     selected recall [default value is '1,2,4,8']
--batch_size                  train batch size [default value is 128]
--num_epochs                  train epoch number [default value is 20]
```

### Test Model
```
python test.py --retrieval_num 10
optional arguments:
--query_img_name              query image name [default value is '/home/data/car/uncropped/008055.jpg']
--data_base                   queried database [default value is 'car_uncropped_resnet18_512_data_base.pth']
--retrieval_num               retrieval number [default value is 8]
```

## Benchmarks
The models are trained on one NVIDIA Tesla V100 (32G) GPU with 20 epochs, 
the learning rate is decayed by 10 on 12th and 16th epoch.

### Model Parameters and FLOPs (Params | FLOPs)
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
      <td align="center">11.80M | 4.29G</td>
      <td align="center">11.81M | 4.29G</td>
      <td align="center">49.85M | 10.69G</td>
      <td align="center">15.80M | 4.30G</td>
    </tr>
    <tr>
      <td align="center">ResNet50</td>
      <td align="center">25.71M | 10.64G</td>
      <td align="center">25.71M | 10.64G</td>
      <td align="center">49.85M | 10.69G</td>
      <td align="center">29.70M | 10.65G</td>
    </tr>
  </tbody>
</table>

### CARS196 (Uncropped | Cropped)
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
      <td align="center">84.9% | 91.0%</td>
      <td align="center">90.9% | 95.2%</td>
      <td align="center">94.9% | 97.1%</td>
      <td align="center">97.1% | 98.5%</td>
      <td align="center"><a href="https://pan.baidu.com/s/1W3-QKVe5HpCAHJTgxI1M5Q">r3sn</a> | <a href="https://pan.baidu.com/s/171Wqa-1TNquzedjlFhaYGg">sf5s</a></td>
    </tr>
    <tr>
      <td align="center">ResNet50</td>
      <td align="center">88.8% | 93.5%</td>
      <td align="center">93.7% | 96.4%</td>
      <td align="center">96.2% | 97.8%</td>
      <td align="center">97.8% | 98.7%</td>
      <td align="center"><a href="https://pan.baidu.com/s/1W3-QKVe5HpCAHJTgxI1M5Q">r3sn</a> | <a href="https://pan.baidu.com/s/171Wqa-1TNquzedjlFhaYGg">sf5s</a></td>
    </tr>
  </tbody>
</table>

### CUB200 (Uncropped | Cropped)
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
      <td align="center">61.2% | 69.5%</td>
      <td align="center">72.6% | 79.1%</td>
      <td align="center">81.7% | 86.8%</td>
      <td align="center">88.3% | 92.0%</td>
      <td align="center"><a href="https://pan.baidu.com/s/1_Ij-bYHZC31cxEWUnYwqwQ">2cfi</a> | <a href="https://pan.baidu.com/s/1deaYb2RWHikztHHsbJyuNw">pi4q</a></td>
    </tr>
    <tr>
      <td align="center">ResNet50</td>
      <td align="center">66.2% | 74.3%</td>
      <td align="center">77.3% | 82.5%</td>
      <td align="center">85.5% | 89.1%</td>
      <td align="center">91.2% | 93.5%</td>
      <td align="center"><a href="https://pan.baidu.com/s/1_Ij-bYHZC31cxEWUnYwqwQ">2cfi</a> | <a href="https://pan.baidu.com/s/1deaYb2RWHikztHHsbJyuNw">pi4q</a></td>
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
      <td align="center">79.3%</td>
      <td align="center">90.6%</td>
      <td align="center">95.8%</td>
      <td align="center">98.6%</td>
      <td align="center"><a href="https://pan.baidu.com/s/17I2nQMK5XBXL1XhiZ2elAg">qgsn</a></td>
    </tr>
    <tr>
      <td align="center">ResNet50</td>
      <td align="center">79.3%</td>
      <td align="center">90.6%</td>
      <td align="center">95.8%</td>
      <td align="center">98.6%</td>
      <td align="center"><a href="https://pan.baidu.com/s/17I2nQMK5XBXL1XhiZ2elAg">qgsn</a></td>
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
      <td align="center">83.6%</td>
      <td align="center">95.7%</td>
      <td align="center">97.1%</td>
      <td align="center">97.7%</td>
      <td align="center">98.1%</td>
      <td align="center">98.4%</td>
      <td align="center"><a href="https://pan.baidu.com/s/10Ow0JhXzRcPVsv5-j14ZjQ">8jmp</a></td>
    </tr>
    <tr>
      <td align="center">ResNet50</td>
      <td align="center">83.6%</td>
      <td align="center">95.7%</td>
      <td align="center">97.1%</td>
      <td align="center">97.7%</td>
      <td align="center">98.1%</td>
      <td align="center">98.4%</td>
      <td align="center"><a href="https://pan.baidu.com/s/10Ow0JhXzRcPVsv5-j14ZjQ">8jmp</a></td>
    </tr>
  </tbody>
</table>

## Results

### CAR/CUB (Uncropped | Cropped)

![CAR/CUB](results/car_cub.png)

### SOP/ISC

![SOP/ISC](results/sop_isc.png)