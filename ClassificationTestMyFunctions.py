import torch
from CP.Classification import SplitCP
from ResNet import ResNet
from utils import get_scores, get_normalize_layer, calibration, prediction, evaluate_predictions
import Score_Functions as scores

import gc
import numpy as np
import seaborn as sns
from tqdm.auto import tqdm
import random
import torch
import torchvision
import os
import pickle
import sys
import argparse
from torchvision import transforms, datasets
from torch.utils.data.dataset import random_split
from sklearn.model_selection import train_test_split
import pandas as pd
import sys 

stdoutOrigin=sys.stdout 


parser = argparse.ArgumentParser(description='Experiments')
parser.add_argument('-a', '--alpha', default=0.1, type=float, help='Desired nominal marginal coverage')
parser.add_argument('-s', '--splits', default=5, type=int, help='Number of experiments to estimate coverage')
parser.add_argument('--dataset', default='CIFAR100', type=str, help='Dataset to be used: CIFAR100, CIFAR10, ImageNet')
parser.add_argument('--batch_size', default=1024, type=int, help='Number of images to send to gpu at once')
parser.add_argument('--coverage_on_label', action='store_true', help='True for getting coverage and size for each label')
parser.add_argument('-PS', '--PlatScale', default = True, action=argparse.BooleanOptionalAction, help='True for getting coverage and size for each label')
parser.add_argument('--seed', metavar='SEED', help='random seed', default=0)

args = parser.parse_args()
sys.stdout = open(str(args.dataset) + "_log.txt", "w")

print(f"args = {args}")
np.random.seed(seed=args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
random.seed(args.seed)

def Find_Acc(device, model, TorchLoader, n_test):
    CorrectSum = 0
    with torch.no_grad():
        for data, labels in TorchLoader:
            pred_labels = torch.argmax(model(data.to(device)), dim = 1).cpu()
            CorrectSum += torch.sum(pred_labels == labels)
    return (CorrectSum/n_test)*100



def TestMyFunctions(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load datasets
    if args.dataset == "CIFAR10":
        # Load train set
        train_dataset = torchvision.datasets.CIFAR10(root='./datasets/',
                                                    train=True,
                                                    transform=torchvision.transforms.ToTensor(),
                                                    download=True)
        # load test set
        test_dataset = torchvision.datasets.CIFAR10(root='./datasets',
                                                    train=False,
                                                    transform=torchvision.transforms.ToTensor())
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                            batch_size=args.batch_size,
                                            shuffle=False)
        
        n_test = 10000
        num_of_classes = 10
        model_path = './CIFAR10_ResNet110_Robust_sigma_0.0.pth.tar'
        model = ResNet(depth=110, num_classes=num_of_classes)
        state = torch.load(model_path, map_location=device)
        normalize_layer = get_normalize_layer("cifar10")
        model = torch.nn.Sequential(normalize_layer, model)
        model.load_state_dict(state['state_dict'])
    elif args.dataset == "CIFAR100":
        # Load train set
        train_dataset = torchvision.datasets.CIFAR100(root='./datasets/',
                                                    train=True,
                                                    transform=torchvision.transforms.ToTensor(),
                                                    download=True)
        # load test set
        test_dataset = torchvision.datasets.CIFAR100(root='./datasets',
                                                    train=False,
                                                    transform=torchvision.transforms.ToTensor())
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                            batch_size=args.batch_size,
                                            shuffle=False)

        n_test = 10000
        num_of_classes = 100
        model_path = './ResNet110_Robust_sigma_0.0.pth.tar' 
        model = ResNet(depth=110, num_classes=num_of_classes)
        state = torch.load(model_path, map_location=device)
        normalize_layer = get_normalize_layer("cifar10")
        model = torch.nn.Sequential(normalize_layer, model)
        model.load_state_dict(state['state_dict'])    

    elif args.dataset == "ImageNet":
        # get dir of imagenet validation set
        imagenet_dir = "./imagenet_val"

        # ImageNet images pre-processing
        transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std= [0.229, 0.224, 0.225])
                ])        
        # load dataset
        test_dataset = datasets.ImageFolder(imagenet_dir, transform)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                            batch_size=args.batch_size,
                                            shuffle=False)
        n_test = 50000
        num_of_classes = 1000
        model = torchvision.models.resnet18(pretrained=True, progress=True)
        print(f"len data = {len(test_dataset)}")

    else:
        print("No such dataset")
        exit(1)


    CalibrationScores = ['APS']
    scores_list = []
    for score in CalibrationScores:
        if score == 'HPS':
            scores_list.append(scores.class_probability_score)
        elif score == 'APS':
            scores_list.append(scores.generalized_inverse_quantile_score)
        elif score == 'RAPS':
            scores_list.append(scores.rank_regularized_score)
        else:
            print("Undefined score function")
            exit(1)
    print(f"scores_list = {scores_list}")



    path = './Results/Dataset_' + str(args.dataset) + '/alpha_' + str(args.alpha) 
    if not os.path.exists(path):
        os.makedirs(path)

    

    # send model to device
    model.to(device)

    # put model in evaluation mode
    model.eval()

    Acc1 = Find_Acc(device, model, test_loader, n_test)
    print(f"Top 1 accuracy = {Acc1}")

    cp = SplitCP(model, CalibrationScores, args.alpha, args.splits, False, '\SavePath', device, 0.7, 1024, args.PlatScale)
    
    
    results = cp.VanillaCP(test_loader, num_of_classes)
    torch.save(results, path + '/APS.pt')

    results = cp.RegularizeCP(test_loader, num_of_classes, k_reg = None, lamda = None)
    torch.save(results, path + '/RAPS.pt')

    results_R_LCP, results_LCP = cp.DistanceLCP(test_loader, num_of_classes)
    torch.save(results_R_LCP, path + '/NCP(RAPS).pt')
    torch.save(results_LCP, path + '/NCP(APS).pt')




TestMyFunctions(args)