import torch
import torch.nn as nn
import torchvision
import os
import torchvision.transforms as tfs
import numpy as np
from tqdm import tqdm
import sys
import wandb

wandb.init(project="residual-connections")

#my files
from architectures.ResNet import ResNet50
from architectures.ViT import ViT

def args_to_dict(args):
    d = {}
    for arg in args:
        k, v = arg.split('=')
        k = k.lstrip('-')
        d[k] = v
    return dict(d)

if __name__ == "__main__":
    args_dict = args_to_dict(sys.argv[1:])

wandb.log(args_dict)

#device
device = torch.device('cuda')

#model definition
assert args_dict['model'] == 'ResNet' or args_dict['model'] == 'ViT'
assert args_dict['zero_weights'] == '0' or args_dict['zero_weights'] == '1'
assert args_dict['include_residual'] == '0' or args_dict['include_residual'] == '1'
if args_dict['model'] == 'ResNet':
    model = ResNet50(include_residual=bool(int(args_dict['include_residual'])), zero_weights=bool(int(args_dict['zero_weights'])))
    #hyperparams - best for resnet
    lr = 0.0005
    grad_clip = 0.1
    weight_decay = 1e-4 #was accidentally 0
    batch_size = 64
    epochs = 25
    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
else:
    model = ViT(include_residual=bool(int(args_dict['include_residual'])), zero_weights=bool(int(args_dict['zero_weights'])))
    lr = 0.00005
    grad_clip = 0.1
    weight_decay = 1e-3
    batch_size = 64
    epochs = 150
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
model = model.to(device).train()

#dataset
transforms = tfs.Compose([tfs.ToTensor(),
                         tfs.RandomCrop(32,padding=3,padding_mode='reflect'),
                         tfs.RandomHorizontalFlip(p=0.5)])
ds = torchvision.datasets.CIFAR100(root='../working/cifar100', transform=transforms,download=True)
train_ds, val_ds = torch.utils.data.random_split(ds, [int(len(ds)*(.9)), len(ds) - int(len(ds)*(.9))])

train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
val_dl = torch.utils.data.DataLoader(val_ds, batch_size=batch_size)

def accuracy(targs, yb):
    asdf, idxs = torch.max(targs, dim=1)
    return torch.sum(idxs == yb).item() / idxs.shape[0]

#training
loss_fn = nn.CrossEntropyLoss()
sched = torch.optim.lr_scheduler.OneCycleLR(optim, lr, epochs=epochs, steps_per_epoch=len(train_dl))


for epoch in range(epochs):
    train_acc = []
    val_acc = []
    train_loss = []
    vals_loss = []
    print('Training')
    for i, (xb, yb) in enumerate(tqdm(train_dl)):
        xb = xb.to(device)
        yb = yb.to(device)
        preds = model(xb)
        loss = loss_fn(preds, yb)
        optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_value_(model.parameters(), grad_clip)
        optim.step()
        optim.zero_grad()
        train_loss.append(loss.item())
        sched.step()
        with torch.no_grad():
            train_acc.append(accuracy(preds, yb))
    
    print('Validating')
    with torch.no_grad():
        for i, (xb, yb) in enumerate(tqdm(val_dl)):
            xb = xb.to(device)
            yb = yb.to(device)
            preds = model(xb)
            val_loss = loss_fn(preds, yb)
            vals_loss.append(val_loss.item())
            val_acc.append(accuracy(preds, yb))
    
    print('epoch', (epoch + 1), 'of', epochs)
    print('train_loss:', np.mean(train_loss), 'accuracy:', np.mean(train_acc))
    wandb.log({"train_loss": np.mean(train_loss)}); wandb.log({"accuracy": np.mean(train_acc)})
    print( 'val_loss:', np.mean(vals_loss), 'val_accuracy:', np.mean(val_acc))
    wandb.log({"val_loss": np.mean(vals_loss)}); wandb.log({"val_accuracy": np.mean(val_acc)})