import random

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

rgb_mean = {'car': [0.4853, 0.4965, 0.4295], 'cub': [0.4707, 0.4601, 0.4549], 'sop': [0.5807, 0.5396, 0.5044]}
rgb_std = {'car': [0.2237, 0.2193, 0.2568], 'cub': [0.2767, 0.2760, 0.2850], 'sop': [0.2901, 0.2974, 0.3095]}


def get_transform(data_name, data_type):
    normalize = transforms.Normalize(rgb_mean[data_name], rgb_std[data_name])
    if data_type == 'train':
        transform = transforms.Compose([transforms.Resize(int(256 * 1.1)), transforms.RandomCrop(256),
                                        transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize])
    else:
        transform = transforms.Compose(
            [transforms.Resize(256), transforms.CenterCrop(256), transforms.ToTensor(), normalize])
    return transform


# random assign meta class for all classes
def create_id(meta_class_size, num_class):
    multiple = num_class // meta_class_size
    remain = num_class % meta_class_size
    if remain != 0:
        multiple += 1

    idx_all = []
    for _ in range(multiple):
        idx_base = [j for j in range(meta_class_size)]
        random.shuffle(idx_base)
        idx_all += idx_base

    idx_all = idx_all[:num_class]
    random.shuffle(idx_all)
    return idx_all


def load_data(meta_id, idx_to_class, data_dict):
    # balance data for each class
    max_size = 300
    meta_data_dict = {i: [] for i in range(max(meta_id) + 1)}
    for i, c in idx_to_class.items():
        meta_class_id = meta_id[i]
        image_list = data_dict[c]
        if len(image_list) > max_size:
            image_list = random.sample(image_list, max_size)
        meta_data_dict[meta_class_id] += image_list
    return meta_data_dict


class ImageReader(Dataset):

    def __init__(self, data_dict, transform):

        classes = [c for c in sorted(data_dict)]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        self.images, self.labels = [], []
        for label in sorted(data_dict):
            for img in data_dict[label]:
                self.images.append(img)
                self.labels.append(class_to_idx[label])

        self.transform = transform

    def __getitem__(self, index):
        path, target = self.images[index], self.labels[index]
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.images)


def recall(feature_vectors, img_labels, rank, weights):
    num_images = len(img_labels)
    img_labels = torch.tensor(img_labels)
    weights = F.softmax(torch.tensor(weights)).view(-1, 1, 1)
    sim_matrix = feature_vectors.bmm(feature_vectors.permute(0, 2, 1).contiguous())
    sim_matrix = torch.sum(weights * sim_matrix, 0)
    sim_matrix[torch.eye(num_images).byte()] = -1

    idx = sim_matrix.argsort(dim=-1, descending=True)
    acc_list = []
    for r in rank:
        correct = (img_labels[idx[:, 0:r]] == img_labels.unsqueeze(dim=-1)).any(dim=-1).float()
        acc_list.append((torch.sum(correct) / num_images).item())
    return acc_list
