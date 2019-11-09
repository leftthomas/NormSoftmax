import argparse
import os
import shutil

import torch
from PIL import Image, ImageDraw

if __name__ == '__main__':
    fix_data_base = torch.load('results/{}'.format('car_uncropped_fixed_unrandom_layer1_resnet18_48_12_data_base.pth'))
    random_data_base = torch.load(
        'results/{}'.format('car_uncropped_random_unrandom_layer1_resnet18_48_12_data_base.pth'))

    for query_index, image_name in enumerate(fix_data_base['test_images']):
        query_label = torch.tensor(fix_data_base['test_labels'][query_index])
        gallery_labels = torch.tensor(fix_data_base['gallery_labels'])

        fix_query_feature = fix_data_base['test_features'][query_index]
        fix_query_feature = fix_query_feature.view(1, *fix_query_feature.size()).permute(1, 0, 2).contiguous()
        random_query_feature = random_data_base['test_features'][query_index]
        random_query_feature = random_query_feature.view(1, *random_query_feature.size()).permute(1, 0, 2).contiguous()

        fix_gallery_features = fix_data_base['gallery_features']
        fix_gallery_features = fix_gallery_features.permute(1, 2, 0).contiguous()
        random_gallery_features = random_data_base['gallery_features']
        random_gallery_features = random_gallery_features.permute(1, 2, 0).contiguous()

        fix_sim_matrix = fix_query_feature.bmm(fix_gallery_features).mean(dim=0).squeeze(dim=0)
        random_sim_matrix = random_query_feature.bmm(random_gallery_features).mean(dim=0).squeeze(dim=0)
        fix_sim_matrix[query_index] = -1
        random_sim_matrix[query_index] = -1
        fix_idx = fix_sim_matrix.argsort(dim=-1, descending=True)
        random_idx = random_sim_matrix.argsort(dim=-1, descending=True)

        fix_correct_num = 0
        for num, index in enumerate(fix_idx[:8]):
            retrieval_label = gallery_labels[index.item()]
            retrieval_status = (retrieval_label == query_label).item()
            if retrieval_status:
                fix_correct_num += 1

        random_correct_num = 0
        for num, index in enumerate(random_idx[:8]):
            retrieval_label = gallery_labels[index.item()]
            retrieval_status = (retrieval_label == query_label).item()
            if retrieval_status:
                random_correct_num += 1

        if fix_correct_num > random_correct_num and fix_correct_num != 8:
            print(image_name)
