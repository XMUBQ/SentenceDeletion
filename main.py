from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report
import torch.nn as nn
import torch.optim as optim
import random
import time
import numpy as np
from tqdm import tqdm
from utils import *
from paramsparser import args
from data_processing import *
from model_combine import Classifier


def eval(model, data_loader, print_result=True):
    with torch.no_grad():
        acc_pred = []
        acc_true = []
        for idx, (x, y_true, discourse_distribution, meta) in tqdm(enumerate(data_loader),
                                                                   total=len(data_loader)):
            y_true = torch.tensor(y_true).long()
            discourse_distribution = torch.tensor(discourse_distribution)
            if cuda_available:
                y_true = y_true.cuda()
                discourse_distribution = discourse_distribution.cuda()

            del_pred, disc_pred = model(x, discourse_distribution)
            _, y_pred = torch.max(del_pred, 1)

            acc_true.extend(y_true.cpu().tolist())
            acc_pred.extend(y_pred.cpu().tolist())
        if print_result:
            print("MACRO: ", precision_recall_fscore_support(acc_true, acc_pred, average='macro'))
            print("MICRO: ", precision_recall_fscore_support(acc_true, acc_pred, average='micro'))
            print("Confusion Metrics \n",
                  classification_report(acc_true, acc_pred, target_names=['retained', 'deleted'], digits=3))
    return classification_report(acc_true, acc_pred, target_names=['retained', 'deleted'], digits=3, output_dict=True)


if __name__ == '__main__':

    discourse_path = 'data/discourse_profile/'
    train_path = 'data/train/'
    test_path = 'data/test/'
    val_path = 'data/val/'
    train_data = get_data(train_path)
    test_data = get_data(test_path)
    val_data = get_data(val_path)
    random.Random(42).shuffle(train_data)

    model_name = args.basemodel

    config = {'num_layers': 1, 'hidden_dim': 512, 'bidirectional': True, 'embedding_dim': 768,
              'dropout': 0.5, 'method': args.basemodel}
    model = Classifier(config)
    if cuda_available:
        model = model.cuda()

    load_epoch = int(args.loadepoch)
    if load_epoch>=0:
        model.load_state_dict(
            torch.load('checkpoints/' + model_name + "_" + str(load_epoch) + ".pt", map_location=device))
    model = model.eval()

    if config['method']=='joint_learning':
        gamma = args.gamma
        model_name += '_' + str(gamma)

    params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.AdamW(params, lr=3e-4, eps=1e-8, weight_decay=0)

    del_loss_function = nn.CrossEntropyLoss(weight=torch.tensor([1.0,1.5]).cuda())
    disc_loss_function = nn.MSELoss()
    m = nn.Softmax(dim=1)
    epochs = args.T

    test_mode = args.testmode
    if not test_mode:
        print(model_name)
        for epoch in range(load_epoch+1, epochs):
            model.train()
            print("EPOCH -- {}".format(epoch))
            for idx, (x, y_true, discourse_distribution, meta) in tqdm(enumerate(train_data), total=len(train_data)):
                optimizer.zero_grad()

                y_true = torch.tensor(y_true).long()
                discourse_distribution = torch.tensor(discourse_distribution)
                if cuda_available:
                    y_true = y_true.cuda()
                    discourse_distribution = discourse_distribution.cuda()

                del_pred, disc_pred = model(x, discourse_distribution)
                if config['method']=='joint_learning':
                    del_loss = del_loss_function(del_pred, y_true)
                    disc_loss = disc_loss_function(m(disc_pred), m(discourse_distribution))
                    final_loss = del_loss + disc_loss * gamma
                    final_loss.backward()
                else:
                    del_loss = del_loss_function(del_pred, y_true)
                    del_loss.backward()

                optimizer.step()

            if epoch >=3:       # models training steady after 3 epochs
                model.eval()
                print("Eval on val")
                eval(model, val_data)
                print("Eval on test")
                eval(model, test_data)
                torch.save(model.state_dict(), 'checkpoints/' + model_name + "_" + str(epoch) + ".pt")
    else:
        load_epoch = args.testepoch
        model.load_state_dict(
            torch.load('checkpoints/' + model_name + "_" + str(load_epoch) + ".pt", map_location=device))
        model = model.eval()

        print(model_name)
        if test_mode == 1:
            output_dir = ''
            data_loader = test_data
        else:
            output_dir = '_val'
            data_loader = val_data

        eval(model, data_loader=data_loader)
