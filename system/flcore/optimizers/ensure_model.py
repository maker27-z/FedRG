#!/usr/bin/env python
import copy
import warnings
import logging

from flcore.trainmodel.models import *
from flcore.trainmodel.transformer import *
from flcore.trainmodel.alexnet import *


logger = logging.getLogger()
logger.setLevel(logging.ERROR)

warnings.simplefilter("ignore")
torch.manual_seed(0)

vocab_size = 98635
max_len = 200
emb_dim = 32
#3*32*32  1600
#3*64*64
def ensure_model( args, id):
    if(args.hete == "Hete2"):
        hete_model = [
            'FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=args.dim, feature_dim=args.feature_dim)',
            'torchvision.models.resnet18(pretrained=False, num_classes=args.num_classes)',
        ]
    elif (args.hete == "Hete3"):
        hete_model = [
            'FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=args.dim, feature_dim=args.feature_dim)',
            'DNN(args.input_dim, mid_dim = args.feature_dim, num_classes=args.num_classes)',
            'alexnet(pretrained=False, num_classes=args.num_classes)'
        ]
    elif (args.hete == "Hete4"):
        hete_model = [
            'FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=args.dim, feature_dim=args.feature_dim)',
            'torchvision.models.googlenet(pretrained=False, aux_logits=False, num_classes=args.num_classes)',
            'mobilenet_v2(pretrained=False, num_classes=args.num_classes)',
            'torchvision.models.resnet18(pretrained=False, num_classes=args.num_classes)'
        ]
    elif (args.hete == "Hete6"):
        hete_model = [
            'FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=args.dim, feature_dim=args.feature_dim)',
            'alexnet(pretrained=False, num_classes=args.num_classes)',
            'torchvision.models.googlenet(pretrained=False, aux_logits=False, num_classes=args.num_classes)',
            'mobilenet_v2(pretrained=False, num_classes=args.num_classes)',
            'torchvision.models.resnet18(pretrained=False, num_classes=args.num_classes)',
            'torchvision.models.resnet34(pretrained=False, num_classes=args.num_classes)',
        ]
    elif (args.hete == "Hete8"):
        hete_model = [
            'FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=args.dim, feature_dim=args.feature_dim)',
            'torchvision.models.googlenet(pretrained=False, aux_logits=False, num_classes=args.num_classes)',
            'mobilenet_v2(pretrained=False, num_classes=args.num_classes)',
            'torchvision.models.resnet18(pretrained=False, num_classes=args.num_classes)',
            'torchvision.models.resnet34(pretrained=False, num_classes=args.num_classes)',
            'torchvision.models.resnet50(pretrained=False, num_classes=args.num_classes)',
            'torchvision.models.resnet101(pretrained=False, num_classes=args.num_classes)',
            'torchvision.models.resnet152(pretrained=False, num_classes=args.num_classes)'
        ]
    else:
        hete_model = [
            'resnet4(num_classes=args.num_classes)',
            'resnet6(num_classes=args.num_classes)',
            'resnet8(num_classes=args.num_classes)',
            'resnet10(num_classes=args.num_classes)',
            'torchvision.models.resnet18(pretrained=False, num_classes=args.num_classes)',
            'torchvision.models.resnet34(pretrained=False, num_classes=args.num_classes)',
            'torchvision.models.resnet50(pretrained=False, num_classes=args.num_classes)',
            'torchvision.models.resnet101(pretrained=False, num_classes=args.num_classes)',
            'torchvision.models.resnet152(pretrained=False, num_classes=args.num_classes)',
        ]
    return BaseHeadSplit(args, hete_model, id)