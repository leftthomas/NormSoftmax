import os

import torch
from PIL import Image
from scipy.io import loadmat
from tqdm import tqdm


def read_txt(path, data_num):
    data = {}
    for line in open(path, 'r', encoding='utf-8'):
        if data_num == 2:
            data_1, data_2 = line.split()
        else:
            data_1, data_2, data_3, data_4, data_5 = line.split()
            data_2 = [data_2, data_3, data_4, data_5]
        data[data_1] = data_2
    return data


def process_car_data(data_path, data_type):
    if not os.path.exists('{}/{}'.format(data_path, data_type)):
        os.mkdir('{}/{}'.format(data_path, data_type))
    train_images, test_images = {}, {}
    annotations = loadmat('{}/cars_annos.mat'.format(data_path))['annotations'][0]
    for img in tqdm(annotations, desc='process {} data for car dataset'.format(data_type)):
        img_name, img_label = str(img[0][0]), str(img[5][0][0])
        if data_type == 'uncropped':
            img = Image.open('{}/{}'.format(data_path, img_name)).convert('RGB')
        else:
            x1, y1, x2, y2 = int(img[1][0][0]), int(img[2][0][0]), int(img[3][0][0]), int(img[4][0][0])
            img = Image.open('{}/{}'.format(data_path, img_name)).convert('RGB').crop((x1, y1, x2, y2))
        img.save('{}/{}/{}'.format(data_path, data_type, os.path.basename(img_name)))
        if int(img_label) < 99:
            if img_label in train_images:
                train_images[img_label].append('{}/{}/{}'.format(data_path, data_type, os.path.basename(img_name)))
            else:
                train_images[img_label] = ['{}/{}/{}'.format(data_path, data_type, os.path.basename(img_name))]
        else:
            if img_label in test_images:
                test_images[img_label].append('{}/{}/{}'.format(data_path, data_type, os.path.basename(img_name)))
            else:
                test_images[img_label] = ['{}/{}/{}'.format(data_path, data_type, os.path.basename(img_name))]
    torch.save({'train': train_images, 'test': test_images}, '{}/{}_data_dicts.pth'.format(data_path, data_type))


def process_cub_data(data_path, data_type):
    if not os.path.exists('{}/{}'.format(data_path, data_type)):
        os.mkdir('{}/{}'.format(data_path, data_type))
    images = read_txt('{}/images.txt'.format(data_path), 2)
    labels = read_txt('{}/image_class_labels.txt'.format(data_path), 2)
    bounding_boxes = read_txt('{}/bounding_boxes.txt'.format(data_path), 5)
    train_images, test_images = {}, {}
    for img_id, img_name in tqdm(images.items(), desc='process {} data for cub dataset'.format(data_type)):
        if data_type == 'uncropped':
            img = Image.open('{}/images/{}'.format(data_path, img_name)).convert('RGB')
        else:
            x1, y1 = int(float(bounding_boxes[img_id][0])), int(float(bounding_boxes[img_id][1]))
            x2, y2 = x1 + int(float(bounding_boxes[img_id][2])), y1 + int(float(bounding_boxes[img_id][3]))
            img = Image.open('{}/images/{}'.format(data_path, img_name)).convert('RGB').crop((x1, y1, x2, y2))
        img.save('{}/{}/{}'.format(data_path, data_type, os.path.basename(img_name)))
        if int(labels[img_id]) < 101:
            if labels[img_id] in train_images:
                train_images[labels[img_id]].append('{}/{}/{}'.format(data_path, data_type, os.path.basename(img_name)))
            else:
                train_images[labels[img_id]] = ['{}/{}/{}'.format(data_path, data_type, os.path.basename(img_name))]
        else:
            if labels[img_id] in test_images:
                test_images[labels[img_id]].append('{}/{}/{}'.format(data_path, data_type, os.path.basename(img_name)))
            else:
                test_images[labels[img_id]] = ['{}/{}/{}'.format(data_path, data_type, os.path.basename(img_name))]
    torch.save({'train': train_images, 'test': test_images}, '{}/{}_data_dicts.pth'.format(data_path, data_type))


def process_sop_data(data_path):
    if not os.path.exists('{}/{}'.format(data_path, 'uncropped')):
        os.mkdir('{}/{}'.format(data_path, 'uncropped'))
    train_images, test_images = {}, {}
    for index, line in enumerate(open('{}/Ebay_train.txt'.format(data_path), 'r', encoding='utf-8')):
        if index != 0:
            _, label, _, img_name = line.split()
            img = Image.open('{}/{}'.format(data_path, img_name)).convert('RGB')
            img.save('{}/{}/{}'.format(data_path, 'uncropped', os.path.basename(img_name)))
            if label in train_images:
                train_images[label].append('{}/{}/{}'.format(data_path, 'uncropped', os.path.basename(img_name)))
            else:
                train_images[label] = ['{}/{}/{}'.format(data_path, 'uncropped', os.path.basename(img_name))]

    for index, line in enumerate(open('{}/Ebay_test.txt'.format(data_path), 'r', encoding='utf-8')):
        if index != 0:
            _, label, _, img_name = line.split()
            img = Image.open('{}/{}'.format(data_path, img_name)).convert('RGB')
            img.save('{}/{}/{}'.format(data_path, 'uncropped', os.path.basename(img_name)))
            if label in test_images:
                test_images[label].append('{}/{}/{}'.format(data_path, 'uncropped', os.path.basename(img_name)))
            else:
                test_images[label] = ['{}/{}/{}'.format(data_path, 'uncropped', os.path.basename(img_name))]
    torch.save({'train': train_images, 'test': test_images}, '{}/{}_data_dicts.pth'.format(data_path, 'uncropped'))


def process_isc_data(data_path):
    if not os.path.exists('{}/{}'.format(data_path, 'uncropped')):
        os.mkdir('{}/{}'.format(data_path, 'uncropped'))
    # train_images, test_images = {}, {}
    # for index, line in enumerate(open('{}/Ebay_train.txt'.format(data_path), 'r', encoding='utf-8')):
    #     if index != 0:
    #         _, label, _, img_name = line.split()
    #         img = Image.open('{}/{}'.format(data_path, img_name)).convert('RGB')
    #         img.save('{}/{}/{}'.format(data_path, 'uncropped', os.path.basename(img_name)))
    #         if label in train_images:
    #             train_images[label].append('{}/{}/{}'.format(data_path, 'uncropped', os.path.basename(img_name)))
    #         else:
    #             train_images[label] = ['{}/{}/{}'.format(data_path, 'uncropped', os.path.basename(img_name))]
    #
    # for index, line in enumerate(open('{}/Ebay_test.txt'.format(data_path), 'r', encoding='utf-8')):
    #     if index != 0:
    #         _, label, _, img_name = line.split()
    #         img = Image.open('{}/{}'.format(data_path, img_name)).convert('RGB')
    #         img.save('{}/{}/{}'.format(data_path, 'uncropped', os.path.basename(img_name)))
    #         if label in test_images:
    #             test_images[label].append('{}/{}/{}'.format(data_path, 'uncropped', os.path.basename(img_name)))
    #         else:
    #             test_images[label] = ['{}/{}/{}'.format(data_path, 'uncropped', os.path.basename(img_name))]
    # torch.save({'train': train_images, 'test': test_images}, '{}/{}_data_dicts.pth'.format(data_path, 'uncropped'))
    # TODO


if __name__ == '__main__':
    process_car_data('data/car', 'uncropped')
    process_car_data('data/car', 'cropped')
    process_cub_data('data/cub', 'uncropped')
    process_cub_data('data/cub', 'cropped')
    print('processing sop dataset')
    process_sop_data('data/sop')
    print('processing isc dataset')
    process_isc_data('data/isc')
