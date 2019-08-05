import argparse

import torch
from PIL import Image
from torchvision import transforms

from utils import rgb_mean, rgb_std

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Image Retrieval Demo')
    parser.add_argument('--query_img_name', default='data/car/uncropped/000001.jpg', type=str, help='query image name')
    parser.add_argument('--data_base', default='car_uncropped_resnet18_48_12_data_base.pth', type=str,
                        help='queried database')
    parser.add_argument('--retrieval_num', default=8, type=int, help='retrieval number')

    opt = parser.parse_args()

    QUERY_IMG_NAME, DATA_BASE, RETRIEVAL_NUM = opt.query_img_name, opt.data_base, opt.retrieval_num
    DATA_NAME = DATA_BASE.split('_')[0]
    normalize = transforms.Normalize(rgb_mean[DATA_NAME], rgb_std[DATA_NAME])
    transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor(), normalize])

    query_data_base = torch.load('results/{}'.format(DATA_BASE))
    if QUERY_IMG_NAME not in query_data_base['test_images']:
        raise FileNotFoundError('{} not found'.format(QUERY_IMG_NAME))
    query_image = transform(Image.open(QUERY_IMG_NAME).convert('RGB'))
    query_label = query_data_base['test_labels'][query_data_base['test_images'].index(QUERY_IMG_NAME)]
    query_feature = query_data_base['test_features'][query_data_base['test_images'].index(QUERY_IMG_NAME)]

    gallery_images = query_data_base['gallery_images']
    gallery_labels = query_data_base['gallery_labels']
    gallery_features = query_data_base['gallery_features']

    num_features = len(feature_labels)
    feature_labels = torch.tensor(feature_labels)
    feature_vectors = feature_vectors.permute(1, 0, 2).contiguous()
    if gallery_vectors is None:
        gallery_vectors = feature_vectors.permute(0, 2, 1).contiguous()
    else:
        gallery_vectors = gallery_vectors.permute(1, 2, 0).contiguous()
    sim_matrix = feature_vectors.bmm(gallery_vectors)
    sim_matrix = torch.mean(sim_matrix, dim=0)
    if gallery_labels is None:
        sim_matrix[torch.eye(num_features).byte()] = -1
        gallery_labels = feature_labels
    else:
        gallery_labels = torch.tensor(gallery_labels)

    idx = sim_matrix.argsort(dim=-1, descending=True)
    acc_list = []
    for r in rank:
        correct = (gallery_labels[idx[:, 0:r]] == feature_labels.unsqueeze(dim=-1)).any(dim=-1).float()
        acc_list.append((torch.sum(correct) / num_features).item())
