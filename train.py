from data import CustomDataset
from model import MFEncoder
from run import config
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import random

random.seed(42)

def universe(config: dict) -> None:

    # dataset
    fname = '5_core_ratings_Musical_Instruments.csv'
    full_dataset = CustomDataset(fname)
    num_user, num_item = full_dataset.num_user, full_dataset.num_item
    train_dataset, val_dataset, test_dataset = full_dataset.split()
    dist_matrix = full_dataset.dist_matrix
    train_loader, val_loader, test_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True),\
        DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=True),\
        DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=True)
    
    # model
    model = MFEncoder(num_user, num_item, config['emb_size'])
    model = model.cuda()

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=1e-6)
    criterion = nn.CrossEntropyLoss()

    # train
    for epoch in range(config['num_epoch']):
        for data in train_loader:
            uid, iid = data[0], data[1]
            # mat_x, mat_y = [[i] for i in data[0]], iid + num_user
            # matrix = dist_matrix[mat_x, mat_y]
            neg_idx = list(range(len(iid)))
            random.shuffle(neg_idx)
            neg_iid = iid[neg_idx]
            neg_label = torch.Tensor(dist_matrix[uid, neg_iid+num_user].reshape(-1,))
            label = torch.cat((torch.ones(len(iid)), neg_label))
            new_label = []
            for i in label:
                if int(i) == 1:
                    new_label.append(0)
                elif int(i) == 3:
                    new_label.append(1)
                else:
                    new_label.append(2)
            label = torch.Tensor(new_label).long()
            uid, iid = torch.cat((uid, uid)), torch.cat((iid, neg_iid))
            uid, iid, label = uid.cuda(), iid.cuda(), label.cuda()

            model.train()
            optimizer.zero_grad()
            output = model(uid, iid)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
        
        model.eval()
        # train validation
        counts, neg_counts, neg_tol = 0, 0, 0
        for data in train_loader:
            uid, iid = data[0], data[1]
            neg_idx = list(range(len(iid)))
            random.shuffle(neg_idx)
            neg_iid = iid[neg_idx]
            neg_label = torch.Tensor(dist_matrix[uid, neg_iid+num_user].reshape(-1,)).cuda()
            uid, iid, neg_iid = uid.cuda(), iid.cuda(), neg_iid.cuda()
            output = model(uid, iid)
            _, pred = torch.max(output.data, 1)
            counts += len(pred[pred == 0])

            output = model(uid, neg_iid)
            _, pred = torch.max(output.data, 1)
            neg_counts += (pred == neg_label).sum().data
            neg_tol += len(pred)
        recall = counts / len(train_dataset)
        neg_recall = neg_counts / neg_tol
        print(f"Recall in train set: {recall:.4f}")
        print(f"Recall in train set of negtive samples: {neg_recall:.4f}")




    return

if __name__ == '__main__':
    universe(config)