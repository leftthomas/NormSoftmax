import argparse

import pandas as pd
import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
import numpy as np
from data_utils import write_json

from model import Model
from utils import ImageReader, recall

# python train.py --data_name car --recalls 1,2,4,8,10,100,1000 --batch_size 12 --num_epochs 20 --ensemble_size 48 --meta_class_size 12 --crop_type cropped --gpu_ids 0,1,2 --model_type resnext50_32x4d --extra_name 0806best

def train(net, optim):
    net.train()
    l_data, t_data, n_data, train_progress = 0, 0, 0, tqdm(train_data_loader)
    for inputs, labels, names in train_progress:
        optim.zero_grad()
        out = net(inputs.to(device_ids[0]))
        loss = cel_criterion(out.permute(0, 2, 1).contiguous(), labels.to(device_ids[0]))
        loss.backward()
        optim.step()
        pred = torch.argmax(out, dim=-1)
        l_data += loss.item()
        t_data += torch.sum((pred.cpu() == labels).float()).item() / ENSEMBLE_SIZE
        n_data += len(labels)
        train_progress.set_description(
            'Epoch {}/{} - Loss:{:.4f} - Acc:{:.2f}%'.format(epoch, NUM_EPOCHS, l_data / n_data, t_data / n_data * 100))
    results['train_loss'].append(l_data / n_data)
    results['train_accuracy'].append(t_data / n_data * 100)


def eval(net, recalls):
    net.eval()
    with torch.no_grad():
        test_features = []
        for inputs, labels, names in test_data_loader:
            out = net(inputs.to(device_ids[0]))
            out = F.normalize(out, dim=-1)
            test_features.append(out.cpu())
        test_features = torch.cat(test_features, dim=0)
        if DATA_NAME == 'isc':
            gallery_features = []
            for inputs, labels in gallery_data_loader:
                out = net(inputs.to(device_ids[0]))
                out = F.normalize(out, dim=-1)
                gallery_features.append(out.cpu())
            gallery_features = torch.cat(gallery_features, dim=0)

    if DATA_NAME == 'isc':
        acc_list, idx = recall(test_features, test_data_set.labels, recalls, gallery_features, gallery_data_set.labels)
    else:
        acc_list, idx = recall(test_features, test_data_set.labels, recalls)
    desc = ''
    for index, recall_id in enumerate(recalls):
        desc += 'R@{}:{:.2f}% '.format(recall_id, acc_list[index] * 100)
        results['test_recall@{}'.format(recall_ids[index])].append(acc_list[index] * 100)
    print(desc)
    global best_recall
    data_base = {}
    if acc_list[0] > best_recall:
        best_recall = acc_list[0]
        data_base['test_images'] = test_data_set.images
        data_base['test_labels'] = test_data_set.labels
        data_base['test_features'] = test_features
        data_base['gallery_images'] = gallery_data_set.images if DATA_NAME == 'isc' else test_data_set.images
        data_base['gallery_labels'] = gallery_data_set.labels if DATA_NAME == 'isc' else test_data_set.labels
        data_base['gallery_features'] = gallery_features if DATA_NAME == 'isc' else test_features
<<<<<<< HEAD
        torch.save(model.state_dict(), 'epochs/{}_{}_{}_model.pth'.format(DATA_NAME, CROP_TYPE, MODEL_TYPE))
        torch.save(data_base, 'results/{}_{}_{}_{}_{}_data_base.pth'.format(DATA_NAME, CROP_TYPE, MODEL_TYPE,
                                                                            ENSEMBLE_SIZE, META_CLASS_SIZE))
    if DATA_NAME == 'car' or DATA_NAME == 'cub':
        np.save('statistics/{}_{}_{}_results_{}.npy'.format(DATA_NAME, CROP_TYPE, MODEL_TYPE, EXTRA_NAME),
                (idx[::, :8]).numpy())
    elif DATA_NAME == 'sop':
        np.save('statistics/{}_{}_{}_results_{}.npy'.format(DATA_NAME, CROP_TYPE, MODEL_TYPE, EXTRA_NAME),
                (idx[::, :1000]).numpy())
    elif DATA_NAME == 'isc':
        np.save('statistics/{}_{}_{}_results_{}.npy'.format(DATA_NAME, CROP_TYPE, MODEL_TYPE, EXTRA_NAME),
                (idx[::, :50]).numpy())
=======
        torch.save(model.state_dict(), 'epochs/{}_{}_{}_model.pth'
                   .format(DATA_NAME, CROP_TYPE, MODEL_TYPE + '_se' if WITH_SE else MODEL_TYPE))
        torch.save(data_base, 'results/{}_{}_{}_{}_{}_data_base.pth'
                   .format(DATA_NAME, CROP_TYPE, MODEL_TYPE + '_se' if WITH_SE else MODEL_TYPE, ENSEMBLE_SIZE,
                           META_CLASS_SIZE))
>>>>>>> 99a22075fb0acaa77b7c145f66b8762fcdb38f59


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Image Retrieval Model')
    parser.add_argument('--data_name', default='car', type=str, choices=['car', 'cub', 'sop', 'isc'],
                        help='dataset name')
    parser.add_argument('--crop_type', default='uncropped', type=str, choices=['uncropped', 'cropped'],
                        help='crop data or not, it only works for car or cub dataset')
    parser.add_argument('--recalls', default='1,2,4,8,10,20,30,40,50,100,1000', type=str, help='selected recall')
    parser.add_argument('--model_type', default='resnet18', type=str,
                        choices=['resnet18', 'resnet34', 'resnet50', 'resnext50_32x4d'], help='backbone type')
    parser.add_argument('--with_se', default='yes', type=str, choices=['yes', 'no'], help='use se block or not')
    parser.add_argument('--batch_size', default=8, type=int, help='train batch size')
    parser.add_argument('--num_epochs', default=20, type=int, help='train epoch number')
    parser.add_argument('--ensemble_size', default=48, type=int, help='ensemble model size')
    parser.add_argument('--meta_class_size', default=12, type=int, help='meta class size')
<<<<<<< HEAD
    parser.add_argument('--gpu_ids', default='0,1,2', type=str, help='selected gpu')
    parser.add_argument('--extra_name', default='train', type=str, help='extra name of file')
=======
    parser.add_argument('--gpu_ids', default='0,1', type=str, help='selected gpu')
>>>>>>> 99a22075fb0acaa77b7c145f66b8762fcdb38f59

    opt = parser.parse_args()

    DATA_NAME, RECALLS, BATCH_SIZE, NUM_EPOCHS = opt.data_name, opt.recalls, opt.batch_size, opt.num_epochs
    ENSEMBLE_SIZE, META_CLASS_SIZE, CROP_TYPE = opt.ensemble_size, opt.meta_class_size, opt.crop_type
<<<<<<< HEAD
    GPU_IDS, MODEL_TYPE, EXTRA_NAME = opt.gpu_ids, opt.model_type, opt.extra_name
=======
    GPU_IDS, MODEL_TYPE, WITH_SE = opt.gpu_ids, opt.model_type, opt.with_se
    WITH_SE = True if WITH_SE == 'yes' else False
>>>>>>> 99a22075fb0acaa77b7c145f66b8762fcdb38f59
    recall_ids, device_ids = [int(k) for k in RECALLS.split(',')], [int(gpu) for gpu in GPU_IDS.split(',')]
    if len(device_ids) != 2:
        raise NotImplementedError('make sure gpu_ids contains two devices')

    results = {'train_loss': [], 'train_accuracy': []}
    for index, recall_id in enumerate(recall_ids):
        results['test_recall@{}'.format(recall_ids[index])] = []

    train_data_set = ImageReader(DATA_NAME, 'train', CROP_TYPE, ENSEMBLE_SIZE, META_CLASS_SIZE)
    train_data_loader = DataLoader(train_data_set, BATCH_SIZE, shuffle=True, num_workers=8)
    test_data_set = ImageReader(DATA_NAME, 'query' if DATA_NAME == 'isc' else 'test', CROP_TYPE)
    test_data_loader = DataLoader(test_data_set, BATCH_SIZE, shuffle=False, num_workers=8)
    if DATA_NAME == 'isc':
        gallery_data_set = ImageReader(DATA_NAME, 'gallery', CROP_TYPE)
        gallery_data_loader = DataLoader(gallery_data_set, BATCH_SIZE, shuffle=False, num_workers=8)

    model = Model(META_CLASS_SIZE, ENSEMBLE_SIZE, MODEL_TYPE, WITH_SE, device_ids)
    print("# trainable parameters:", sum(param.numel() if param.requires_grad else 0 for param in model.parameters()))
    optimizer = Adam(model.parameters(), lr=1e-4)
    lr_scheduler = MultiStepLR(optimizer, milestones=[int(NUM_EPOCHS * 0.5), int(NUM_EPOCHS * 0.7)], gamma=0.1)
    cel_criterion = CrossEntropyLoss()

    # write img index to json file
    l_data, t_data, n_data, name_progress = 0, 0, 0, tqdm(test_data_loader)
    name_list, label_list, name_index = [], [], {}
    for inputs, labels, names in name_progress:
        for i in list(names):
            name_list.append(i)
        for i in list(labels):
            label_list.append(int(i))
    print('The length of the list is: ' + str(len(name_list)))
    for i, name in enumerate(name_list):
        name_index[i] = {}
        name_index[i]['label'] = str(label_list[i])
        name_index[i]['name'] = str(name)
    write_json(name_index, 'statistics/{}_name_index.json'.format(DATA_NAME))

    best_recall = 0
    for epoch in range(1, NUM_EPOCHS + 1):
        train(model, optimizer)
        lr_scheduler.step(epoch)
        eval(model, recall_ids)
        # save statistics
        data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
<<<<<<< HEAD
        data_frame.to_csv('statistics/{}_{}_{}_results_{}.csv'.format(DATA_NAME, CROP_TYPE, MODEL_TYPE, EXTRA_NAME),
=======
        data_frame.to_csv('statistics/{}_{}_{}_results.csv'
                          .format(DATA_NAME, CROP_TYPE, MODEL_TYPE + '_se' if WITH_SE else MODEL_TYPE),
>>>>>>> 99a22075fb0acaa77b7c145f66b8762fcdb38f59
                          index_label='epoch')
