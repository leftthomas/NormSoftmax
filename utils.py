import random

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

rgb_mean = {'car': [0.4853, 0.4965, 0.4295], 'cub': [0.4707, 0.4601, 0.4549], 'sop': [0.5807, 0.5396, 0.5044]}
rgb_std = {'car': [0.2237, 0.2193, 0.2568], 'cub': [0.2767, 0.2760, 0.2850], 'sop': [0.2901, 0.2974, 0.3095]}


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


class ImageReader(Dataset):

    def __init__(self, data_name, data_type, crop_type='uncropped', ensemble_size=None, meta_class_size=None):
        if crop_type == 'cropped' and data_name not in ['car', 'cub']:
            raise NotImplementedError('cropped data only works for car or cub dataset')

        data_dict = torch.load('data/{}/{}_data_dicts.pth'.format(data_name, crop_type))[data_type]
        class_to_idx = dict(zip(sorted(data_dict), range(len(data_dict))))
        normalize = transforms.Normalize(rgb_mean[data_name], rgb_std[data_name])
        if data_type == 'train':
            self.transform = transforms.Compose([transforms.Resize((256, 256)), transforms.RandomHorizontalFlip(),
                                                 transforms.ToTensor(), normalize])
            meta_ids = [create_id(meta_class_size, len(data_dict)) for _ in range(ensemble_size)]
            # balance data for each class
            max_size = 300
            self.images, self.labels = [], []
            for label, image_list in data_dict.items():
                if len(image_list) > max_size:
                    image_list = random.sample(image_list, max_size)
                for img in image_list:
                    self.images.append(img)
                    meta_label = []
                    for meta_id in meta_ids:
                        meta_label.append(meta_id[class_to_idx[label]])
                    meta_label = torch.tensor(meta_label)
                    self.labels.append(meta_label)
        else:
            self.transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor(), normalize])
            self.images, self.labels = [], []
            for label, image_list in data_dict.items():
                for img in image_list:
                    self.images.append(img)
                    self.labels.append(class_to_idx[label])

    def __getitem__(self, index):
        path, target = self.images[index], self.labels[index]
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.images)


def recall(feature_vectors, img_labels, rank):
    num_images = len(img_labels)
    img_labels = torch.tensor(img_labels)
    feature_vectors = feature_vectors.permute(1, 0, 2).contiguous()
    sim_matrix = feature_vectors.bmm(feature_vectors.permute(0, 2, 1).contiguous())
    sim_matrix = torch.sum(sim_matrix, dim=0)
    sim_matrix[torch.eye(num_images).byte()] = -1

    idx = sim_matrix.argsort(dim=-1, descending=True)
    acc_list = []
    for r in rank:
        correct = (img_labels[idx[:, 0:r]] == img_labels.unsqueeze(dim=-1)).any(dim=-1).float()
        acc_list.append((torch.sum(correct) / num_images).item())
    return acc_list
