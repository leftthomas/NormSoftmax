import argparse
import copy

import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data.dataloader import DataLoader

from model import Model
from utils import ImageReader, create_id, get_transform, load_data, recall


def train(net, data_dict, optim):
    net.train()
    data_set = ImageReader(data_dict, get_transform(DATA_NAME, 'train'))
    data_loader = DataLoader(data_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)

    l_data, t_data, n_data = 0.0, 0, 0
    for inputs, labels in data_loader:
        optim.zero_grad()
        out = net(inputs.to(DEVICE))
        loss = cel_criterion(out, labels.to(DEVICE))
        print('loss:{:.4f}'.format(loss.item()), end='\r')
        loss.backward()
        optim.step()
        _, pred = torch.max(out, 1)
        l_data += loss.item()
        t_data += torch.sum(pred.cpu() == labels).item()
        n_data += len(labels)

    return l_data / n_data, t_data / n_data


def eval(net, data_dict, ensemble_num, recalls):
    net.eval()
    data_set = ImageReader(data_dict, get_transform(DATA_NAME, 'test'))
    data_loader = DataLoader(data_set, BATCH_SIZE, shuffle=False, num_workers=8)

    features = []
    with torch.no_grad():
        for inputs, labels in data_loader:
            out = net(inputs.to(DEVICE))
            out = F.normalize(out, dim=-1)
            features.append(out.cpu())
    features = torch.cat(features, 0)
    torch.save(features, 'results/{}_test_features_{:03}.pth'.format(DATA_NAME, ensemble_num))
    # load feature vectors
    features = [torch.load('results/{}_test_features_{:03}.pth'.format(DATA_NAME, d)) for d in
                range(1, ensemble_num + 1)]
    features = torch.stack(features, 0)
    acc_list = recall(features, data_set.labels, recalls)
    desc = ''
    for index, recall_id in enumerate(recalls):
        desc += 'R@{}:{:.2f}% '.format(recall_id, acc_list[index] * 100)
    print(desc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Image Retrieval Model')
    parser.add_argument('--data_name', default='car', type=str, choices=['car', 'cub', 'sop'], help='dataset name')
    parser.add_argument('--recalls', default='1,2,4,8', type=str, help='selected recall')
    parser.add_argument('--batch_size', default=128, type=int, help='train batch size')
    parser.add_argument('--num_epochs', default=20, type=int, help='train epoch number')
    parser.add_argument('--ensemble_size', default=12, type=int, help='ensemble model size')
    parser.add_argument('--meta_class_size', default=12, type=int, help='meta class size')

    opt = parser.parse_args()

    DATA_NAME, RECALLS, BATCH_SIZE, NUM_EPOCHS = opt.data_name, opt.recalls, opt.batch_size, opt.num_epochs
    ENSEMBLE_SIZE, META_CLASS_SIZE = opt.ensemble_size, opt.meta_class_size
    recall_ids = [int(k) for k in RECALLS.split(',')]
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_dicts = torch.load('data/{}/data_dicts.pth'.format(DATA_NAME))
    train_data, test_data = data_dicts['train'], data_dicts['test']
    # sort classes and fix the class order
    all_class = sorted(train_data)
    idx_to_class = {i: all_class[i] for i in range(len(all_class))}

    for i in range(1, ENSEMBLE_SIZE + 1):
        print('Training ensemble #{}'.format(i))
        meta_id = create_id(META_CLASS_SIZE, len(data_dicts['train']))
        meta_data_dict = load_data(meta_id, idx_to_class, train_data)
        model = Model(META_CLASS_SIZE).to(DEVICE)
        optimizer = Adam(model.parameters(), lr=1e-4)
        lr_scheduler = MultiStepLR(optimizer, milestones=[int(NUM_EPOCHS * 0.5), int(NUM_EPOCHS * 0.7)], gamma=0.1)
        cel_criterion = CrossEntropyLoss()

        best_acc, best_model = 0, None
        for epoch in range(1, NUM_EPOCHS + 1):
            lr_scheduler.step(epoch)
            train_loss, train_acc = train(model, meta_data_dict, optimizer)
            print('Epoch {}/{} - Loss:{:.4f} - Acc:{:.4f}'.format(epoch, NUM_EPOCHS, train_loss, train_acc))
            # deep copy the model
            if train_acc > best_acc:
                best_acc = train_acc
                best_model = copy.deepcopy(model)
                torch.save(model.state_dict(), 'epochs/{}_model_{:03}.pth'.format(DATA_NAME, i))
        eval(best_model, test_data, i, recall_ids)
