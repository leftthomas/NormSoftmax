import argparse

import pandas as pd
import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from model import Model
from utils import ImageReader, recall


def train(net, data_loader, optim):
    net.train()
    l_data, t_data, n_data, train_progress = 0, 0, 0, tqdm(data_loader)
    for inputs, labels in train_progress:
        optim.zero_grad()
        out = net(inputs.to(device_ids[0]))
        loss = cel_criterion(out.permute(0, 2, 1).contiguous(), labels.to(device_ids[0]))
        loss.backward()
        optim.step()
        pred = torch.argmax(out, dim=-1)
        l_data += loss.item()
        t_data += torch.sum(pred.cpu() == labels).item() / ENSEMBLE_SIZE
        n_data += len(labels)
        train_progress.set_description(
            'Epoch {}/{} - Loss:{:.4f} - Acc:{:.2f}%'.format(epoch, NUM_EPOCHS, l_data / n_data, t_data / n_data * 100))
    results['train_loss'].append(l_data / n_data)
    results['train_accuracy'].append(t_data / n_data * 100)
    global best_acc
    if t_data / n_data > best_acc:
        best_acc = t_data / n_data
        torch.save(model.state_dict(), 'epochs/{}_model.pth'.format(DATA_NAME))


def eval(net, data_loader, recalls):
    net.eval()
    features = []
    with torch.no_grad():
        for inputs, labels in data_loader:
            out = net(inputs.to(device_ids[0]))
            out = F.normalize(out, dim=-1)
            features.append(out.cpu())
    features = torch.cat(features, dim=0)
    torch.save(features, 'results/{}_test_features.pth'.format(DATA_NAME))
    acc_list = recall(features, test_data_set.labels, recalls)
    desc = ''
    for index, recall_id in enumerate(recalls):
        desc += 'R@{}:{:.2f}% '.format(recall_id, acc_list[index] * 100)
        results['test_recall@{}'.format(recall_ids[index])].append(acc_list[index] * 100)
    print(desc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Image Retrieval Model')
    parser.add_argument('--data_name', default='car', type=str, choices=['car', 'cub', 'sop'], help='dataset name')
    parser.add_argument('--crop_type', default='uncropped', type=str, choices=['uncropped', 'cropped'],
                        help='crop data or not, it only works for car or cub dataset')
    parser.add_argument('--recalls', default='1,2,4,8', type=str, help='selected recall')
    parser.add_argument('--batch_size', default=16, type=int, help='train batch size')
    parser.add_argument('--num_epochs', default=20, type=int, help='train epoch number')
    parser.add_argument('--ensemble_size', default=48, type=int, help='ensemble model size')
    parser.add_argument('--meta_class_size', default=12, type=int, help='meta class size')
    parser.add_argument('--gpu_ids', default='0,1', type=str, help='selected gpu')

    opt = parser.parse_args()

    DATA_NAME, RECALLS, BATCH_SIZE, NUM_EPOCHS = opt.data_name, opt.recalls, opt.batch_size, opt.num_epochs
    ENSEMBLE_SIZE, META_CLASS_SIZE, CROP_TYPE = opt.ensemble_size, opt.meta_class_size, opt.crop_type
    GPU_IDS = opt.gpu_ids
    recall_ids, device_ids = [int(k) for k in RECALLS.split(',')], [int(gpu) for gpu in GPU_IDS.split(',')]
    if len(device_ids) != 2:
        raise NotImplementedError('make sure gpu_ids contains two devices')

    results = {'train_loss': [], 'train_accuracy': []}
    for index, recall_id in enumerate(recall_ids):
        results['test_recall@{}'.format(recall_ids[index])] = []

    train_data_set = ImageReader(DATA_NAME, 'train', CROP_TYPE, ENSEMBLE_SIZE, META_CLASS_SIZE)
    train_data_loader = DataLoader(train_data_set, BATCH_SIZE, shuffle=True, num_workers=8)
    test_data_set = ImageReader(DATA_NAME, 'test', CROP_TYPE)
    test_data_loader = DataLoader(test_data_set, BATCH_SIZE, shuffle=False, num_workers=8)

    model = Model(META_CLASS_SIZE, ENSEMBLE_SIZE, device_ids)
    print("# trainable parameters:", sum(param.numel() if param.requires_grad else 0 for param in model.parameters()))
    optimizer = Adam(model.parameters(), lr=1e-4)
    lr_scheduler = MultiStepLR(optimizer, milestones=[int(NUM_EPOCHS * 0.5), int(NUM_EPOCHS * 0.7)], gamma=0.1)
    cel_criterion = CrossEntropyLoss()

    best_acc = 0
    for epoch in range(1, NUM_EPOCHS + 1):
        train(model, train_data_loader, optimizer)
        lr_scheduler.step(epoch)
        eval(model, test_data_loader, recall_ids)
        # save statistics
        data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
        data_frame.to_csv('statistics/{}_results.csv'.format(DATA_NAME), index_label='epoch')
