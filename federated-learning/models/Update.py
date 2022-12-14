#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import torch
import torchvision.datasets
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets,transforms
from tqdm import tqdm
import torch.nn.functional as F
import models.teacher_model as tm
from utils.distillation import distilling
from models.test import test_img

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


class LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
        # self.dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))

    def train(self, stu_net):
        teach_model = tm.ConvNet(args=self.args)
        teach_model.to(self.args.device)

        path ='models/teacher_model_statedict/teach_model.pth'
        teach_model.load_state_dict(torch.load(path))
        # #知识蒸馏 student—>teacher
        T = 8
        lr = 2e-4
        distil_epoch = 2
        state_dict, loss = distilling(stu_net, teach_model, T, lr, distil_epoch, self.ldr_train)
        teach_model.load_state_dict(state_dict)

        optimizer = torch.optim.SGD(teach_model.parameters(), lr=0.001, momentum=self.args.momentum)
        for i in range(self.args.local_ep):
            tm.train(teach_model, self.args.device, self.ldr_train, optimizer, i+1)
        # 模型剪枝
        # print('\npruning 20%')

        mask = tm.filter_prune(teach_model, 20)
        teach_model.set_masks(mask)
        # _, acc = tm.test(teach_model, self.args.device, self.Test_loader)
        tm.train(teach_model, self.args.device, self.ldr_train, optimizer, 1)

        torch.save(teach_model.state_dict(), path)
        teach_net = teach_model
        # 知识蒸馏开始 teacher->student
        T = 8
        lr = 2e-4
        distil_epoch = 6
        state_dict, loss = distilling(teach_net, stu_net, T, lr, distil_epoch, self.ldr_train)


        return state_dict, loss
