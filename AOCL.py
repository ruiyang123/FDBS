import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import copy
import utils
import argparse

import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader


parser = argparse.ArgumentParser("./AOCL.py", description="Adversarial Continual Learning Experiment")

# Dataset
parser.add_argument("--factor", type=str, default="Numbers_DIL")

# Training Methods
parser.add_argument("--lwf",action="store_true")
parser.add_argument("--autolwf", action="store_true")
parser.add_argument("--using-feature-output", action="store_true")
parser.add_argument("--using-classification-output", action="store_true")
parser.add_argument("--aocl", action = "store_true")

# Training Regime
parser.add_argument('--iters', type=int, default=3000)
parser.add_argument('--batch-size', type=int, default=32)
parser.add_argument('--task-num', type=int, default=4)
parser.add_argument('--lamb', type=float, default = 0.2)

class FeatureExtractor(nn.Module):

    def __init__(self):
        super(FeatureExtractor, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        x = self.conv(x).squeeze()
        return x


class LabelPredictor(nn.Module):

    def __init__(self):
        super(LabelPredictor, self).__init__()

        self.layer = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.ReLU(),

            nn.Linear(512, 10),
        )

    def forward(self, h):
        c = self.layer(h)
        return c


class DomainClassifier(nn.Module):

    def __init__(self,input_features_number):
        super(DomainClassifier, self).__init__()

        self.layer = nn.Sequential(
            nn.Linear(input_features_number,input_features_number),
            nn.BatchNorm1d(input_features_number),
            nn.ReLU(),

            nn.Linear(input_features_number, input_features_number),
            nn.BatchNorm1d(input_features_number),
            nn.ReLU(),

            nn.Linear(input_features_number, input_features_number),
            nn.BatchNorm1d(input_features_number),
            nn.ReLU(),

            nn.Linear(input_features_number, input_features_number),
            nn.BatchNorm1d(input_features_number),
            nn.ReLU(),

            nn.Linear(input_features_number, 1),
        )

    def forward(self, h):
        y = self.layer(h)
        return y



def validate_onetask_aocl(feature_ex, label_pre, dataset, batch_size=32, test_size=1024):
    feature_ex.eval()
    label_pre.eval()
    data_loader =DataLoader(dataset,batch_size=batch_size,shuffle=True)
    total_tested = total_correct = 0

    for data, labels in data_loader:

        if test_size:
            if total_tested >= test_size:
                break
        data, labels = data.cuda(), labels.cuda()

        with torch.no_grad():
            predictions = label_pre(feature_ex(data))
            _, predicted = torch.max(predictions,1)
        total_correct += (predicted == labels).sum().item()
        total_tested += len(data)
    print(total_tested, total_correct)

    precision = total_correct/total_tested

    feature_ex.train()
    label_pre.train()

    return total_correct, total_tested, precision


def validate_alltask_aocl(feature_ex, label_pre, datasets, batch_size=32, test_size=1024):

    precs ={}

    for task_id, dataset in enumerate(datasets):
        total_correct, total_tested, precision = validate_onetask_aocl(feature_ex, label_pre, dataset, batch_size=batch_size, test_size=test_size)
        precs["t{}".format(task_id)] = precision

        print("task: {}, {}/{}, {}".format(task_id, total_correct, total_tested, precision))
    return precs




def aocl_training(args):

    df_prec = pd.DataFrame()

    features_dim = 0

    if args.using_classification_output:
        features_dim += 10
    if args.using_feature_output:
        features_dim += 512
    print(features_dim)

    # Define Models
    feature_extractor = FeatureExtractor().cuda()
    label_predictor = LabelPredictor().cuda()
    domain_classifier = DomainClassifier(features_dim).cuda()

    class_criterion = nn.CrossEntropyLoss()
    domain_criterion = nn.BCEWithLogitsLoss()

    optimizer_F = optim.Adam(feature_extractor.parameters())
    optimizer_C = optim.Adam(label_predictor.parameters())
    optimizer_D = optim.Adam(domain_classifier.parameters())


    # Load Dataset
    with open(args.factor + '.pk', 'rb') as f:
        ((train_datasets, test_datasets), config, classes_per_task) = pickle.load(f)

    # Training Process
    for t in range(args.task_num):
        if t == 0:
            iters = 2500
        else:
            iters = args.iters
        train_loader = DataLoader(train_datasets[t], batch_size=args.batch_size, shuffle=True)

        iters_left = 1

        running_loss = 0
        running_D_loss = 0

        if t == 0:
            for batch_index in range(1,iters+1):

                iters_left -=1
                if iters_left == 0:
                    train_iter = iter(train_loader)
                    iters_left = len(train_iter)
                x,y = next(train_iter)
                x, y = x.cuda(), y.cuda()

                features = feature_extractor(x)
                predictions = label_predictor(features)

                loss = class_criterion(predictions,y)

                optimizer_F.zero_grad()
                optimizer_C.zero_grad()
                loss.backward()
                optimizer_F.step()
                optimizer_C.step()

                running_loss +=loss.item()

                if batch_index %500 ==0:
                    print("Task : %d"%(t))
                    print("batch : %d , loss : %.4f"%(batch_index,running_loss/500))
                    running_loss = 0
        else:
            previous_fe = copy.deepcopy(feature_extractor)
            previous_cl = copy.deepcopy(label_predictor)

            for batch_index in range(1,iters+1):

                iters_left -=1
                if iters_left == 0:
                    train_iter = iter(train_loader)
                    iters_left = len(train_iter)
                x,y = next(train_iter)
                x, y = x.cuda(), y.cuda()


                for k in range(1):
                    features0 = previous_fe(x)
                    features1 = feature_extractor(x)

                    if len(features0.size()) == 1:
                        continue

                    if args.using_classification_output:
                        class_output0 = previous_cl(features0)
                        class_output1 = label_predictor(features1)
                        if not args.using_feature_output:
                            features0 = class_output0
                            features1 = class_output1
                        else:
                            features0 = torch.cat([class_output0,features0],dim=1)
                            features1 = torch.cat([class_output1,features1],dim=1)


                    if batch_index == 0:
                        mixed_features = features0
                        model_label = torch.randint(1,(features0.shape[0],1))
                    else:
                        mixed_features = torch.cat([features0,features1],dim=0)
                        model_label = torch.zeros([features0.shape[0]+features1.shape[0],1]).cuda()
                        model_label[:features0.shape[0]] = 1

                    # train classifier

                    domain_logits = domain_classifier(mixed_features.detach())
                    loss = domain_criterion(domain_logits,model_label)
                    optimizer_D.zero_grad()
                    loss.backward()
                    optimizer_D.step()
                running_D_loss += loss.item()



                # train feature extractor and label classifier

                for k in range(3):
                    features1 = feature_extractor(x)
                    features0 = previous_fe(x)

                    predictions = label_predictor(features1)
                    predictions0 = previous_cl(features0)
                    if len(features1.size()) == 1:
                        continue

                    if args.using_classification_output:
                        if not args.using_feature_output:
                            features1 = predictions
                        else:
                            features1 = torch.cat([predictions, features1],dim=1)


                    if len(features1.size()) == 1:
                        continue
                    domain_logits = domain_classifier(features1)

                    if not args.lwf:
                        loss_lwf = 0
                    else:
                        loss_lwf = utils.loss_fn_kd(predictions, predictions0)

                    if not args.autolwf:
                        loss_autolwf = 0
                    else:
                        loss_autolwf = utils.adjust_loss_fn_kd(predictions, predictions0)


                    loss = - args.lamb* domain_criterion(domain_logits,torch.zeros([features1.shape[0],1]).cuda()) + class_criterion(predictions,y) + loss_lwf + loss_autolwf

                    optimizer_F.zero_grad()
                    optimizer_C.zero_grad()


                    loss.backward()
                    optimizer_F.step()
                    optimizer_C.step()


                running_loss += loss.item()

                if batch_index %300 ==0:
                    print("Task : %d"%(t))
                    print("batch : %d , loss : %.4f , loss D : %.4f"%(batch_index,running_loss/300,running_D_loss/300))
                    running_loss = 0
                    running_D_loss = 0
        # Validation

        print("--------------Evaluation for task %d--------------------"%(t))
        precs = validate_alltask_aocl(feature_extractor, label_predictor, test_datasets, batch_size=32, test_size=2048)
        df_prec = df_prec.append(precs,ignore_index=True)
        print("--------------------END Evaluation -----------------------------")

    save_name = "aocl"
    if args.lwf:
        save_name += "+lwf"
    elif args.autolwf:
        save_name += "+autolwf"
    if args.using_feature_output:
        save_name += "+fo"
    if args.using_classification_output:
        save_name += "co"

    save_name += ".csv"

    df_prec.to_csv(save_name,index=False)





def normal_training(args):
    df_prec = pd.DataFrame()
    features_dim = 0

    if args.using_classification_output:
        features_dim += 10
    if args.using_feature_output:
        features_dim += 512
    print(features_dim)

    # Define Models
    feature_extractor = FeatureExtractor().cuda()
    label_predictor = LabelPredictor().cuda()

    class_criterion = nn.CrossEntropyLoss()

    optimizer_F = optim.Adam(feature_extractor.parameters())
    optimizer_C = optim.Adam(label_predictor.parameters())


    # Load Dataset
    with open(args.factor + '.pk', 'rb') as f:
        ((train_datasets, test_datasets), config, classes_per_task) = pickle.load(f)

    # Training Process
    for t in range(args.task_num):
        if t == 0:
            iters = 1100
        else:
            iters = args.iters
        train_loader = DataLoader(train_datasets[t], batch_size=args.batch_size, shuffle=True)

        iters_left = 1

        running_loss = 0


        if t == 0:
            for batch_index in range(1,iters+1):

                iters_left -=1
                if iters_left == 0:
                    train_iter = iter(train_loader)
                    iters_left = len(train_iter)
                x,y = next(train_iter)
                x, y = x.cuda(), y.cuda()

                features = feature_extractor(x)
                predictions = label_predictor(features)

                loss = class_criterion(predictions,y)

                optimizer_F.zero_grad()
                optimizer_C.zero_grad()
                loss.backward()
                optimizer_F.step()
                optimizer_C.step()

                running_loss +=loss.item()

                if batch_index %500 ==0:
                    print("Task : %d"%(t))
                    print("batch : %d , loss : %.4f"%(batch_index,running_loss/500))
                    running_loss = 0
        else:
            previous_fe = copy.deepcopy(feature_extractor)
            previous_cl = copy.deepcopy(label_predictor)

            for batch_index in range(1,iters+1):

                iters_left -=1
                if iters_left == 0:
                    train_iter = iter(train_loader)
                    iters_left = len(train_iter)
                x,y = next(train_iter)
                x, y = x.cuda(), y.cuda()
                # train feature extractor and label classifier

                for k in range(1):
                    features1 = feature_extractor(x)
                    features0 = previous_fe(x)

                    predictions = label_predictor(features1)
                    predictions0 = previous_cl(features0)


                    if len(features1.size()) == 1:
                        continue

                    if not args.lwf:
                        loss_lwf = 0
                    else:
                        loss_lwf = utils.loss_fn_kd(predictions, predictions0)

                    if not args.autolwf:
                        loss_autolwf = 0
                    else:
                        loss_autolwf = utils.adjust_loss_fn_kd(predictions, predictions0)


                    loss = class_criterion(predictions,y) + loss_lwf + loss_autolwf

                    optimizer_F.zero_grad()
                    optimizer_C.zero_grad()


                    loss.backward()
                    optimizer_F.step()
                    optimizer_C.step()


                running_loss += loss.item()

                if batch_index %300 ==0:
                    print("Task : %d"%(t))
                    print("batch : %d , loss : %.4f "%(batch_index,running_loss/300))
                    running_loss = 0
                    running_D_loss = 0
        # Validation

        print("--------------Evaluation for task %d--------------------"%(t))
        precs = validate_alltask_aocl(feature_extractor, label_predictor, test_datasets, batch_size=32, test_size=2048)
        df_prec = df_prec.append(precs, ignore_index=True)
        print("--------------------END Evaluation -----------------------------")

    save_name = "naive"
    if args.lwf:
        save_name = "+lwf"
    elif args.autolwf:
        save_name = "+autolwf"
    save_name += ".csv"

    df_prec.to_csv(save_name,index=False)

if __name__ == '__main__':
    args = parser.parse_args()

    if args.aocl :
        aocl_training(args)
    else:
        normal_training(args)


