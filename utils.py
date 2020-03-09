import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms

rgb_mean = {'car': [0.4853, 0.4965, 0.4295], 'cub': [0.4707, 0.4601, 0.4549], 'sop': [0.5807, 0.5396, 0.5044],
            'isc': [0.8324, 0.8109, 0.8041]}
rgb_std = {'car': [0.2237, 0.2193, 0.2568], 'cub': [0.2767, 0.2760, 0.2850], 'sop': [0.2901, 0.2974, 0.3095],
           'isc': [0.2206, 0.2378, 0.2444]}


class ImageReader(Dataset):

    def __init__(self, data_path, data_name, data_type, crop_type):
        if crop_type == 'cropped' and data_name not in ['car', 'cub']:
            raise NotImplementedError('cropped data only works for car or cub dataset')

        data_dict = torch.load('{}/{}/{}_data_dicts.pth'.format(data_path, data_name, crop_type))[data_type]
        self.class_to_idx = dict(zip(sorted(data_dict), range(len(data_dict))))
        normalize = transforms.Normalize(rgb_mean[data_name], rgb_std[data_name])
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
