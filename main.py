import torch
import torch.nn as nn
import os
from src.config import arg
from src.resnet32 import resnet32
from src.train import train
from src.evaluate import validate
from src.dataset import train_loader, val_loader
from src.utils import save_checkpoint
from src.swish import swish

def main():

    best_prec1 = 0
    # Check the save_dir exists or not
    if not os.path.exists('save_temp'):
        os.makedirs('save_temp')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    model = resnet32(swish).to(device)
    model.cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), arg.lr,
                                momentum=arg.momentum,
                                weight_decay=arg.weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                        milestones=[100, 150], last_epoch=arg.start_epoch - 1)

    for epoch in range(0, arg.epochs):

        # train for one epoch
        print('Training {} model'.format('resnet32'))
        print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
        train(train_loader(), model, criterion, optimizer, epoch)
        lr_scheduler.step()

        # evaluate on validation set
        prec1 = validate(val_loader(), model, criterion)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)

        if epoch > 0 and epoch % arg.save_every == 0:
            save_checkpoint(model.state_dict(), filename=os.path.join(arg.save_dir, 'checkpoint.th'))
        if is_best:
            save_checkpoint(model.state_dict(), filename=os.path.join(arg.save_dir, 'model.th'))

    return best_prec1