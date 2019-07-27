# SREML
A PyTorch implementation of SREML based on the paper [Squeezed Randomized Ensembles for Metric Learning]().

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
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center">CARS196</td>
      <td align="center">529,365,376</td>
      <td align="center">1,011,079,808</td>
      <td align="center">1,118,974,592</td>
      <td align="center">1,094,093,696</td>
    </tr>
    <tr>
      <td align="center">CUB200</td>
      <td align="center">529,365,376</td>
      <td align="center">1,011,079,808</td>
      <td align="center">1,118,974,592</td>
      <td align="center">1,094,093,696</td>
    </tr>
    <tr>
      <td align="center">SOP</td>
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
      <td align="center">89.85%</td>
      <td align="center">62.22%</td>
      <td align="center">94.33%</td>
      <td align="center">73.48%</td>
      <td align="center">/</td>
      <td align="center">83.88%</td>
    </tr>
    <tr>
      <td align="center">R@2</td>
      <td align="center">93.99%</td>
      <td align="center">73.26%</td>
      <td align="center">96.96%</td>
      <td align="center">82.26%</td>
      <td align="center">/</td>
      <td align="center">89.15%</td>
    </tr>
    <tr>
      <td align="center">R@4</td>
      <td align="center">96.25%</td>
      <td align="center">82.31%</td>
      <td align="center">98.32%</td>
      <td align="center">88.79%</td>
      <td align="center">/</td>
      <td align="center">92.52%</td>
    </tr>
    <tr>
      <td align="center">R@8</td>
      <td align="center">97.88%</td>
      <td align="center">89.52%</td>
      <td align="center">99.07%</td>
      <td align="center">93.11%</td>
      <td align="center">/</td>
      <td align="center">94.99%</td>
    </tr>
    <tr>
      <td align="center">R@10</td>
      <td align="center">98.23%</td>
      <td align="center">90.90%</td>
      <td align="center">99.20%</td>
      <td align="center">94.07%</td>
      <td align="center">/</td>
      <td align="center">95.62%</td>
    </tr>
    <tr>
      <td align="center">R@20</td>
      <td align="center">99.03%</td>
      <td align="center">94.55%</td>
      <td align="center">99.56%</td>
      <td align="center">96.47%</td>
      <td align="center">/</td>
      <td align="center">96.83%</td>
    </tr>
    <tr>
      <td align="center">R@30</td>
      <td align="center">99.30%</td>
      <td align="center">96.40%</td>
      <td align="center">99.67%</td>
      <td align="center">97.33%</td>
      <td align="center">/</td>
      <td align="center">97.46%</td>
    </tr>
    <tr>
      <td align="center">R@40</td>
      <td align="center">99.57%</td>
      <td align="center">97.15%</td>
      <td align="center">99.72%</td>
      <td align="center">98.02%</td>
      <td align="center">/</td>
      <td align="center">97.85%</td>
    </tr>
    <tr>
      <td align="center">R@50</td>
      <td align="center">99.63%</td>
      <td align="center">97.69%</td>
      <td align="center">99.75%</td>
      <td align="center">98.40%</td>
      <td align="center">/</td>
      <td align="center">98.13%</td>
    </tr>
    <tr>
      <td align="center">R@100</td>
      <td align="center">99.84%</td>
      <td align="center">98.82%</td>
      <td align="center">99.85%</td>
      <td align="center">99.24%</td>
      <td align="center">/</td>
      <td align="center">98.72%</td>
    </tr>
    <tr>
      <td align="center">R@1000</td>
      <td align="center">100.00%</td>
      <td align="center">99.98%</td>
      <td align="center">100.00%</td>
      <td align="center">99.97%</td>
      <td align="center">/</td>
      <td align="center">99.74%</td>
    </tr>
  </tbody>
</table>

Here is the recall details of `resnet34` backbone:

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
      <td align="center">91.23%</td>
      <td align="center">64.53%</td>
      <td align="center">95.46%</td>
      <td align="center">74.05%</td>
      <td align="center">/</td>
      <td align="center">/</td>
    </tr>
    <tr>
      <td align="center">R@2</td>
      <td align="center">95.06%</td>
      <td align="center">73.92%</td>
      <td align="center">97.52%</td>
      <td align="center">82.92%</td>
      <td align="center">/</td>
      <td align="center">/</td>
    </tr>
    <tr>
      <td align="center">R@4</td>
      <td align="center">97.15%</td>
      <td align="center">81.89%</td>
      <td align="center">98.52%</td>
      <td align="center">89.10%</td>
      <td align="center">/</td>
      <td align="center">/</td>
    </tr>
    <tr>
      <td align="center">R@8</td>
      <td align="center">98.20%</td>
      <td align="center">88.22%</td>
      <td align="center">99.05%</td>
      <td align="center">93.59%</td>
      <td align="center">/</td>
      <td align="center">/</td>
    </tr>
    <tr>
      <td align="center">R@10</td>
      <td align="center">98.46%</td>
      <td align="center">90.21%</td>
      <td align="center">99.21%</td>
      <td align="center">94.48%</td>
      <td align="center">/</td>
      <td align="center">/</td>
    </tr>
    <tr>
      <td align="center">R@20</td>
      <td align="center">99.08%</td>
      <td align="center">94.06%</td>
      <td align="center">99.58%</td>
      <td align="center">96.59%</td>
      <td align="center">/</td>
      <td align="center">/</td>
    </tr>
    <tr>
      <td align="center">R@30</td>
      <td align="center">99.29%</td>
      <td align="center">95.83%</td>
      <td align="center">99.72%</td>
      <td align="center">97.65%</td>
      <td align="center">/</td>
      <td align="center">/</td>
    </tr>
    <tr>
      <td align="center">R@40</td>
      <td align="center">99.48%</td>
      <td align="center">96.94%</td>
      <td align="center">99.75%</td>
      <td align="center">98.09%</td>
      <td align="center">/</td>
      <td align="center">/</td>
    </tr>
    <tr>
      <td align="center">R@50</td>
      <td align="center">99.52%</td>
      <td align="center">97.50%</td>
      <td align="center">99.78%</td>
      <td align="center">98.36%</td>
      <td align="center">/</td>
      <td align="center">/</td>
    </tr>
    <tr>
      <td align="center">R@100</td>
      <td align="center">99.77%</td>
      <td align="center">98.89%</td>
      <td align="center">99.88%</td>
      <td align="center">99.05%</td>
      <td align="center">/</td>
      <td align="center">/</td>
    </tr>
    <tr>
      <td align="center">R@1000</td>
      <td align="center">100.00%</td>
      <td align="center">99.98%</td>
      <td align="center">100.00%</td>
      <td align="center">99.95%</td>
      <td align="center">/</td>
      <td align="center">/</td>
    </tr>
  </tbody>
</table>

Here is the recall details of `resnet50` backbone:

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
      <td align="center">93.09%</td>
      <td align="center">66.19%</td>
      <td align="center">95.87%</td>
      <td align="center">76.69%</td>
      <td align="center">/</td>
      <td align="center">/</td>
    </tr>
    <tr>
      <td align="center">R@2</td>
      <td align="center">96.11%</td>
      <td align="center">76.38%</td>
      <td align="center">97.74%</td>
      <td align="center">84.45%</td>
      <td align="center">/</td>
      <td align="center">/</td>
    </tr>
    <tr>
      <td align="center">R@4</td>
      <td align="center">97.58%</td>
      <td align="center">84.45%</td>
      <td align="center">98.56%</td>
      <td align="center">89.67%</td>
      <td align="center">/</td>
      <td align="center">/</td>
    </tr>
    <tr>
      <td align="center">R@8</td>
      <td align="center">98.41%</td>
      <td align="center">90.12%</td>
      <td align="center">99.18%</td>
      <td align="center">93.25%</td>
      <td align="center">/</td>
      <td align="center">/</td>
    </tr>
    <tr>
      <td align="center">R@10</td>
      <td align="center">98.67%</td>
      <td align="center">91.81%</td>
      <td align="center">99.26%</td>
      <td align="center">94.41%</td>
      <td align="center">/</td>
      <td align="center">/</td>
    </tr>
    <tr>
      <td align="center">R@20</td>
      <td align="center">99.42%</td>
      <td align="center">95.02%</td>
      <td align="center">99.56%</td>
      <td align="center">96.56%</td>
      <td align="center">/</td>
      <td align="center">/</td>
    </tr>
    <tr>
      <td align="center">R@30</td>
      <td align="center">99.59%</td>
      <td align="center">96.61%</td>
      <td align="center">99.69%</td>
      <td align="center">97.55%</td>
      <td align="center">/</td>
      <td align="center">/</td>
    </tr>
    <tr>
      <td align="center">R@40</td>
      <td align="center">99.69%</td>
      <td align="center">97.52%</td>
      <td align="center">99.74%</td>
      <td align="center">98.19%</td>
      <td align="center">/</td>
      <td align="center">/</td>
    </tr>
    <tr>
      <td align="center">R@50</td>
      <td align="center">99.75%</td>
      <td align="center">98.01%</td>
      <td align="center">99.77%</td>
      <td align="center">98.50%</td>
      <td align="center">/</td>
      <td align="center">/</td>
    </tr>
    <tr>
      <td align="center">R@100</td>
      <td align="center">99.86%</td>
      <td align="center">99.02%</td>
      <td align="center">99.88%</td>
      <td align="center">99.09%</td>
      <td align="center">/</td>
      <td align="center">/</td>
    </tr>
    <tr>
      <td align="center">R@1000</td>
      <td align="center">100.00%</td>
      <td align="center">99.97%</td>
      <td align="center">100.00%</td>
      <td align="center">99.97%</td>
      <td align="center">/</td>
      <td align="center">/</td>
    </tr>
  </tbody>
</table>

Here is the recall details of `resnext50_32x4d` backbone:

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
      <td align="center">93.75%</td>
      <td align="center">69.92%</td>
      <td align="center">96.26%</td>
      <td align="center">78.36%</td>
      <td align="center">/</td>
      <td align="center">/</td>
    </tr>
    <tr>
      <td align="center">R@2</td>
      <td align="center">96.53%</td>
      <td align="center">79.64%</td>
      <td align="center">97.88%</td>
      <td align="center">85.89%</td>
      <td align="center">/</td>
      <td align="center">/</td>
    </tr>
    <tr>
      <td align="center">R@4</td>
      <td align="center">97.80%</td>
      <td align="center">86.61%</td>
      <td align="center">98.72%</td>
      <td align="center">91.09%</td>
      <td align="center">/</td>
      <td align="center">/</td>
    </tr>
    <tr>
      <td align="center">R@8</td>
      <td align="center">98.66%</td>
      <td align="center">91.46%</td>
      <td align="center">99.30%</td>
      <td align="center">94.21%</td>
      <td align="center">/</td>
      <td align="center">/</td>
    </tr>
    <tr>
      <td align="center">R@10</td>
      <td align="center">98.82%</td>
      <td align="center">92.76%</td>
      <td align="center">99.39%</td>
      <td align="center">95.14%</td>
      <td align="center">/</td>
      <td align="center">/</td>
    </tr>
    <tr>
      <td align="center">R@20</td>
      <td align="center">99.24%</td>
      <td align="center">95.61%</td>
      <td align="center">99.53%</td>
      <td align="center">97.01%</td>
      <td align="center">/</td>
      <td align="center">/</td>
    </tr>
    <tr>
      <td align="center">R@30</td>
      <td align="center">99.50%</td>
      <td align="center">96.88%</td>
      <td align="center">99.70%</td>
      <td align="center">97.82%</td>
      <td align="center">/</td>
      <td align="center">/</td>
    </tr>
    <tr>
      <td align="center">R@40</td>
      <td align="center">99.58%</td>
      <td align="center">97.64%</td>
      <td align="center">99.74%</td>
      <td align="center">98.13%</td>
      <td align="center">/</td>
      <td align="center">/</td>
    </tr>
    <tr>
      <td align="center">R@50</td>
      <td align="center">99.63%</td>
      <td align="center">98.06%</td>
      <td align="center">99.80%</td>
      <td align="center">98.41%</td>
      <td align="center">/</td>
      <td align="center">/</td>
    </tr>
    <tr>
      <td align="center">R@100</td>
      <td align="center">99.84%</td>
      <td align="center">99.04%</td>
      <td align="center">99.88%</td>
      <td align="center">99.09%</td>
      <td align="center">/</td>
      <td align="center">/</td>
    </tr>
    <tr>
      <td align="center">R@1000</td>
      <td align="center">100.00%</td>
      <td align="center">99.98%</td>
      <td align="center">100.00%</td>
      <td align="center">99.97%</td>
      <td align="center">/</td>
      <td align="center">/</td>
    </tr>
  </tbody>
</table>

