import argparse

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms

rgb_mean = {'car_uncropped': [0.470, 0.461, 0.456], 'car_cropped': [0.421, 0.402, 0.405],
            'cub_uncropped': [0.484, 0.503, 0.452], 'cub_cropped': [0.462, 0.468, 0.420],
            'sop_uncropped': [0.579, 0.539, 0.505], 'isc_uncropped': [0.832, 0.811, 0.804]}
rgb_std = {'car_uncropped': [0.257, 0.256, 0.262], 'car_cropped': [0.264, 0.260, 0.262],
           'cub_uncropped': [0.176, 0.176, 0.187], 'cub_cropped': [0.206, 0.206, 0.214],
           'sop_uncropped': [0.230, 0.234, 0.234], 'isc_uncropped': [0.204, 0.224, 0.233]}


def get_mean_std(data_path, data_name, crop_type):
    data_dict = torch.load('{}/{}/{}_data_dicts.pth'.format(data_path, data_name, crop_type))['train']

    mean, std, num = np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32), 0.0
    for key, value in data_dict.items():
        for path in value:
            num += 1
            img = np.array(Image.open(path).convert('RGB').resize((224, 224), Image.BILINEAR), dtype=np.float32)
            for i in range(3):
                mean[i] += img[:, :, i].mean()
                std[i] += img[:, :, i].std()
    mean, std = mean / (num * 255), std / (num * 255)
    print('[{}-{}] Mean: {} Std: {}'.format(data_name, crop_type, np.around(mean, decimals=3),
                                            np.around(std, decimals=3)))


class ImageReader(Dataset):

    def __init__(self, data_path, data_name, data_type, crop_type):
        if crop_type == 'cropped' and data_name not in ['car', 'cub']:
            raise NotImplementedError('cropped data only works for car or cub dataset')

        data_dict = torch.load('{}/{}/{}_data_dicts.pth'.format(data_path, data_name, crop_type))[data_type]
        self.class_to_idx = dict(zip(sorted(data_dict), range(len(data_dict))))
        normalize = transforms.Normalize(rgb_mean['{}_{}'.format(data_name, crop_type)],
                                         rgb_std['{}_{}'.format(data_name, crop_type)])
        if data_type == 'train':
            self.transform = transforms.Compose([transforms.Resize((256, 256)), transforms.RandomCrop(224),
                                                 transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize])
        else:
            self.transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), normalize])
        self.images, self.labels, self.weights = [], [], {}
        for label, image_list in data_dict.items():
            self.images += image_list
            self.labels += [self.class_to_idx[label]] * len(image_list)
            self.weights[self.class_to_idx[label]] = len(image_list)
        # class-wise weight for overcome dataset imbalance
        sum_weight = 0.0
        for key, value in self.weights.items():
            self.weights[key] = len(self.labels) / value
            sum_weight += self.weights[key]
        for key, value in self.weights.items():
            self.weights[key] = value / sum_weight * len(self.class_to_idx)

    def __getitem__(self, index):
        path, target = self.images[index], self.labels[index]
        weight = self.weights[target]
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        return img, target, weight

    def __len__(self):
        return len(self.images)


def recall(feature_vectors, feature_labels, rank, gallery_vectors=None, gallery_labels=None):
    num_features = len(feature_labels)
    feature_labels = torch.tensor(feature_labels, device=feature_vectors.device)
    gallery_vectors = feature_vectors if gallery_vectors is None else gallery_vectors

    sim_matrix = torch.mm(feature_vectors, gallery_vectors.t().contiguous())

    if gallery_labels is None:
        sim_matrix.fill_diagonal_(0)
        gallery_labels = feature_labels
    else:
        gallery_labels = torch.tensor(gallery_labels, device=feature_vectors.device)

    idx = sim_matrix.topk(k=rank[-1], dim=-1, largest=True)[1]
    acc_list = []
    for r in rank:
        correct = (gallery_labels[idx[:, 0:r]] == feature_labels.unsqueeze(dim=-1)).any(dim=-1).float()
        acc_list.append((torch.sum(correct) / num_features).item())
    return acc_list


class LabelSmoothingCrossEntropyLoss(nn.Module):
    def __init__(self, smoothing=0.1, temperature=1.0):
        super().__init__()
        self.smoothing = smoothing
        self.temperature = temperature

    def forward(self, x, target, weight):
        log_probs = F.log_softmax(x / self.temperature, dim=-1)
        nll_loss = -log_probs.gather(dim=-1, index=target.unsqueeze(dim=-1)).squeeze(dim=-1)
        smooth_loss = -log_probs.mean(dim=-1)
        loss = (1.0 - self.smoothing) * nll_loss + self.smoothing * smooth_loss
        return (weight * loss).mean()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process datasets')
    parser.add_argument('--data_path', default='/home/data', type=str, help='datasets path')

    opt = parser.parse_args()
    get_mean_std(opt.data_path, 'car', 'uncropped')
    get_mean_std(opt.data_path, 'car', 'cropped')
    get_mean_std(opt.data_path, 'cub', 'uncropped')
    get_mean_std(opt.data_path, 'cub', 'cropped')
    get_mean_std(opt.data_path, 'sop', 'uncropped')
    get_mean_std(opt.data_path, 'isc', 'uncropped')
