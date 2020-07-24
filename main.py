import os
import time
import argparse
import datetime
import json
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import PIL
import mlflow

from src import consts
from src import model
from src import data


def main():
    date = (datetime.datetime.utcnow() + datetime.timedelta(hours=9)).strftime('%y%m%d_%H%M%S')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', type=str, default=os.path.join('logs', date))
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epoch', type=int, default=500)
    parser.add_argument('--loss_log_interval', type=int, default=10)
    parser.add_argument('--model_save_interval', type=int, default=50)
    parser.add_argument('--trunk_model', type=str, default='resnet34')
    parser.add_argument('--final_relu', type=bool, default=False)
    parser.add_argument('--embedding_size', type=int, default=64)
    parser.add_argument('--memo', type=str, default='')
    args = parser.parse_args()

    logdir = args.logdir
    os.makedirs(logdir, exist_ok=True)
    modeldir = os.path.join(logdir, 'models')
    os.makedirs(modeldir, exist_ok=True)

    with open(os.path.join(logdir, 'params.json'), mode='w') as f:
        json.dump(args.__dict__, f, indent=4)

    epoch = args.epoch
    log_loss_interval = args.loss_log_interval

    # dataset =
    transform = transforms.Compose([
                        PIL.Image.fromarray,
                        transforms.Resize([args.img_size, args.img_size]),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                      ])

    dataset = data.WashHandDataset(root='dataset/sliced', cache='.cache/train.db', transform=transform)
    label_num = len(consts.n2l)

    n_samples = len(dataset)
    train_size = int(n_samples * 0.8)
    val_size = n_samples - train_size

    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)

    layer_sizes = [args.embedding_size, label_num]

    classifier = model.Classifier(args.trunk_model, layer_sizes, final_relu=args.final_relu)
    classifier = classifier.to(device)
    print(classifier)

    criterion = torch.nn.CrossEntropyLoss()

    classifier.train()

    optim = torch.optim.Adam(params=[
            {'params': classifier.trunk.parameters(), 'lr': 0.000005},
            {'params': classifier.embedder.parameters(), 'lr': 0.00001}],
            weight_decay=0.0001)
    print(optim)

    # mlflow start
    mlflow.start_run()
    # Log args
    for key, value in vars(args).items():
        mlflow.log_param(key, value)
    # Log other info
    mlflow.log_param('transform', transform)
    mlflow.log_param('optimizer', optim.__repr__())
    mlflow.log_param('embedder', classifier.embedder.__repr__())
    mlflow.log_param('num_data', len(dataset))

    num_iters = len(train_loader)
    start_time = time.time()
    for e in range(epoch):
        running_loss = 0
        classifier.train()
        for i, (img, label) in enumerate(train_loader):
            img = img.to(device)
            label = label.to(device)
            # forward #
            output = classifier(img)

            loss = criterion(output, label)
            running_loss += loss

            optim.zero_grad()
            loss.backward()
            optim.step()

            if (i+1) % log_loss_interval == 0 or i == 0:
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = 'Elapsed [{}], Epoch [{}/{}], Iteration [{}/{}], loss:{: .4f} \
                        '.format(et, e+1, epoch, i+1, num_iters, loss)
                print(log)

        # lossはtensor型なのでスカラで取り出す
        mlflow.log_metric('loss_train', (running_loss/num_iters).cpu().item(), step=e+1)
        classifier.eval()
        loss = 0
        # y_true = []
        # y_pred = []
        for i, (img, label) in enumerate(valid_loader):
            img = img.to(device)
            label = label.to(device)
            with torch.no_grad():
                # forward #
                output = classifier(img)
                loss += criterion(output, label).data
                # y_true.append(label.detach().cpu().numpy())
                # y_pred.append(output.detach().cpu().numpy())
        valid_loss = loss / len(valid_loader)
        print('Elapsed [{}], Epoch [{}/{}], validation loss: {: .4f} \
              '.format(et, e+1, epoch, valid_loss))
        mlflow.log_metric('loss_valid', valid_loss.cpu().item(), step=e+1)

        if (e+1) % args.model_save_interval == 0:
            save_path = os.path.join(modeldir, f'{e+1:05}_classifier.pth')
            torch.save(classifier.state_dict(), save_path)
            print('model saved.')

    mlflow.end_run()


if __name__ == '__main__':
    main()