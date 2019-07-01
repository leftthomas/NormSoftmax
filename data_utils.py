import torch
from scipy.io import loadmat


def read_txt(path):
    data = {}
    for line in open(path, 'r', encoding='utf-8'):
        data_1, data_2 = line.split()
        data[data_1] = data_2
    return data


def process_car_data(data_path):
    train_images, test_images = {}, {}
    annotations = loadmat('{}/cars_annos.mat'.format(data_path))['annotations'][0]
    for img in annotations:
        img_name, img_label = str(img[0][0]), str(img[-2][0][0])
        if int(img_label) < 99:
            if img_label in train_images:
                train_images[img_label].append('{}/{}'.format(data_path, img_name))
            else:
                train_images[img_label] = ['{}/{}'.format(data_path, img_name)]
        else:
            if img_label in test_images:
                test_images[img_label].append('{}/{}'.format(data_path, img_name))
            else:
                test_images[img_label] = ['{}/{}'.format(data_path, img_name)]
    torch.save({'train': train_images, 'test': test_images}, '{}/{}'.format(data_path, data_dicts))


def process_cub_data(data_path):
    images = read_txt('{}/images.txt'.format(data_path))
    labels = read_txt('{}/image_class_labels.txt'.format(data_path))
    train_images, test_images = {}, {}
    for img_id, img_name in images.items():
        if int(labels[img_id]) < 101:
            if labels[img_id] in train_images:
                train_images[labels[img_id]].append('{}/images/{}'.format(data_path, img_name))
            else:
                train_images[labels[img_id]] = ['{}/images/{}'.format(data_path, img_name)]
        else:
            if labels[img_id] in test_images:
                test_images[labels[img_id]].append('{}/images/{}'.format(data_path, img_name))
            else:
                test_images[labels[img_id]] = ['{}/images/{}'.format(data_path, img_name)]
    torch.save({'train': train_images, 'test': test_images}, '{}/{}'.format(data_path, data_dicts))


def process_sop_data(data_path):
    train_images, test_images = {}, {}
    for index, line in enumerate(open('{}/Ebay_train.txt'.format(data_path), 'r', encoding='utf-8')):
        if index != 0:
            _, label, _, img_name = line.split()
            if label in train_images:
                train_images[label].append('{}/{}'.format(data_path, img_name))
            else:
                train_images[label] = ['{}/{}'.format(data_path, img_name)]

    for index, line in enumerate(open('{}/Ebay_test.txt'.format(data_path), 'r', encoding='utf-8')):
        if index != 0:
            _, label, _, img_name = line.split()
            if label in test_images:
                test_images[label].append('{}/{}'.format(data_path, img_name))
            else:
                test_images[label] = ['{}/{}'.format(data_path, img_name)]
    torch.save({'train': train_images, 'test': test_images}, '{}/{}'.format(data_path, data_dicts))


if __name__ == '__main__':
    data_dicts = 'data_dicts.pth'
    process_car_data('data/car')
    process_cub_data('data/cub')
    process_sop_data('data/sop')
