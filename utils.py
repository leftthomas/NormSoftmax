import argparse

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms

rgb_mean = {'car_train_uncropped': [0.470, 0.461, 0.456], 'car_train_cropped': [0.421, 0.402, 0.405],
            'car_test_uncropped': [0.470, 0.458, 0.453], 'car_test_cropped': [0.423, 0.400, 0.402],
            'cub_train_uncropped': [0.484, 0.503, 0.452], 'cub_train_cropped': [0.462, 0.468, 0.420],
            'cub_test_uncropped': [0.488, 0.496, 0.412], 'cub_test_cropped': [0.482, 0.475, 0.392],
            'sop_train_uncropped': [0.579, 0.539, 0.505], 'sop_test_uncropped': [0.582, 0.540, 0.505],
            'isc_train_uncropped': [0.832, 0.811, 0.804], 'isc_query_uncropped': [0.830, 0.808, 0.802],
            'isc_gallery_uncropped': [0.834, 0.813, 0.806]}
rgb_std = {'car_train_uncropped': [0.265, 0.264, 0.269], 'car_train_cropped': [0.270, 0.267, 0.269],
           'car_test_uncropped': [0.268, 0.267, 0.271], 'car_test_cropped': [0.273, 0.269, 0.271],
           'cub_train_uncropped': [0.183, 0.183, 0.194], 'cub_train_cropped': [0.211, 0.211, 0.219],
           'cub_test_uncropped': [0.181, 0.179, 0.191], 'cub_test_cropped': [0.200, 0.198, 0.206],
           'sop_train_uncropped': [0.238, 0.242, 0.242], 'sop_test_uncropped': [0.237, 0.241, 0.242],
           'isc_train_uncropped': [0.209, 0.229, 0.237], 'isc_query_uncropped': [0.211, 0.231, 0.238],
           'isc_gallery_uncropped': [0.211, 0.231, 0.239]}


def get_mean_std(data_path, data_name, data_type, crop_type):
    data_dict = torch.load('{}/{}/{}_data_dicts.pth'.format(data_path, data_name, crop_type))[data_type]

    mean, std, num = np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32), 0.0
    for key, value in data_dict.items():
        for path in value:
            num += 1
            img = np.array(Image.open(path).convert('RGB'), dtype=np.float32)
            for i in range(3):
                mean[i] += img[:, :, i].mean()
                std[i] += img[:, :, i].std()
    mean, std = mean / (num * 255), std / (num * 255)
    print('[{}-{}-{}] Mean: {} Std: {}'.format(data_name, data_type, crop_type, mean, std))


class ImageReader(Dataset):

    def __init__(self, data_path, data_name, data_type, crop_type):
        if crop_type == 'cropped' and data_name not in ['car', 'cub']:
            raise NotImplementedError('cropped data only works for car or cub dataset')

        data_dict = torch.load('{}/{}/{}_data_dicts.pth'.format(data_path, data_name, crop_type))[data_type]
        self.class_to_idx = dict(zip(sorted(data_dict), range(len(data_dict))))
        normalize = transforms.Normalize(rgb_mean['{}_{}_{}'.format(data_name, data_type, crop_type)],
                                         rgb_std['{}_{}_{}'.format(data_name, data_type, crop_type)])
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
    get_mean_std(opt.data_path, 'car', 'train', 'uncropped')
    get_mean_std(opt.data_path, 'car', 'train', 'cropped')
    get_mean_std(opt.data_path, 'car', 'test', 'uncropped')
    get_mean_std(opt.data_path, 'car', 'test', 'cropped')
    get_mean_std(opt.data_path, 'cub', 'train', 'uncropped')
    get_mean_std(opt.data_path, 'cub', 'train', 'cropped')
    get_mean_std(opt.data_path, 'cub', 'test', 'uncropped')
    get_mean_std(opt.data_path, 'cub', 'test', 'cropped')
    get_mean_std(opt.data_path, 'sop', 'train', 'uncropped')
    get_mean_std(opt.data_path, 'sop', 'test', 'uncropped')
    get_mean_std(opt.data_path, 'isc', 'train', 'uncropped')
    get_mean_std(opt.data_path, 'isc', 'query', 'uncropped')
    get_mean_std(opt.data_path, 'isc', 'gallery', 'uncropped')
