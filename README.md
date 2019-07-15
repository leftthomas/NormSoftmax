# DCN
A PyTorch implementation of Diverse Capsule Network based on the paper [Diverse Capsule Network for Image Retrieval]().

## Requirements
- [Anaconda](https://www.anaconda.com/download/)
- [PyTorch](https://pytorch.org)
```
conda install pytorch torchvision -c pytorch
```
- capsule-layer
```
pip install git+https://github.com/leftthomas/CapsuleLayer.git@master
```

## Datasets
[cars196](http://ai.stanford.edu/~jkrause/cars/car_dataset.html), [cub200](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html)
and [sop](http://cvgl.stanford.edu/projects/lifted_struct/) are used in this repo.

You should download these datasets by yourself, and extract them into `data` directory, make sure the dir names are 
`car`, `cub` and `sop`. Then run `data_utils.py` to preprocess them.

## Usage
### Train Model
```
python train.py --num_epochs 30
optional arguments:
--data_name                   dataset name [default value is 'car'](choices=['car', 'cub', 'sop'])
--crop_type                   crop data or not, it only works for car or cub dataset [default value is 'uncropped'](choices=['uncropped', 'cropped'])
--recalls                     selected recall [default value is '1,2,4,8']
--batch_size                  train batch size [default value is 16]
--num_epochs                  train epochs number [default value is 20]
--ensemble_size               ensemble model size [default value is 48]
--meta_class_size             meta class size [default value is 12]
```

## Benchmarks
Adam optimizer is used with learning rate scheduling. The models are trained with batch size of 16 on one 
NVIDIA Tesla V100 (32G) GPU.

The images are preprocessed with random resize, random crop, random horizontal flip, and normalize. 
For `Cars196` and `CUB200` datasets, `20` epochs, ensemble size `48` and meta class size `12` are used. For `SOP` dataset,
`20` epochs, ensemble size `48` and meta class size `500` is used.
Here is the recall details:

<table>
  <thead>
    <tr>
      <th>Dataset</th>
      <th>Cars196</th>
      <th>CUB200</th>
      <th>SOP</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center">R@1</td>
      <td align="center">85.07%</td>
      <td align="center">64.16%</td>
      <td align="center">56.93%</td>
    </tr>
    <tr>
      <td align="center">R@2</td>
      <td align="center">91.13%</td>
      <td align="center">75.22%</td>
      <td align="center">61.68%</td>
    </tr>
    <tr>
      <td align="center">R@4</td>
      <td align="center">94.45%</td>
      <td align="center">83.25%</td>
      <td align="center">66.18%</td>
    </tr>
    <tr>
      <td align="center">R@8</td>
      <td align="center">96.73%</td>
      <td align="center">89.30%</td>
      <td align="center">70.08%</td>
    </tr>
    <tr>
      <td align="center">R@10</td>
      <td align="center">97.42%</td>
      <td align="center">91.17%</td>
      <td align="center">71.32%</td>
    </tr>
    <tr>
      <td align="center">R@100</td>
      <td align="center">99.79%</td>
      <td align="center">99.27%</td>
      <td align="center">82.58%</td>
    </tr>
    <tr>
      <td align="center">R@1000</td>
      <td align="center">100.00%</td>
      <td align="center">99.98%</td>
      <td align="center">91.60%</td>
    </tr>
  </tbody>
</table>


