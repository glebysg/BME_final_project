#!/usr/bin/env python3.6

import sys
import numpy as np
import LSTM
import AlexNet
import GoogLeNet
import torch, torchvision
import matplotlib.pyplot as plt

import torch.nn as nn
from torch.autograd import Variable
plt.switch_backend('agg')

useCNN = True
useGoogLeNet = False
usePreloadModel = False
isTraining = True
withCuda = True
withLSTM = False
net = []
train_obj_name = 'pkls/trainout'
test_obj_name = 'pkls/testout'
model='AlexNet_Regression'
#model='LSTM64_LR_Momentum_fold1'
fignames = 'figures/'+model+'.png'

if(useCNN):
    ###################################33tres
    ##Mock data for CNNs
    print("************ Preparing the data ************")
    ######### Data should be a FloatTensor of dims [1,3,224,224] #########
    ######### Label should be a FloatTensor of dims [1] #########    
    ##To be replaced with the frames from the videos
    # train_loader = [[torch.randn(1,3,224,224),torch.randn(1)], [torch.randn(1,3,224,224),torch.randn(1)], \
    # [torch.randn(1,3,224,224),torch.randn(1)], [torch.randn(1,3,224,224),torch.randn(1)]]

    ##To be replaced with the frames from the videos
    # test_loader = [[torch.randn(1,3,224,224),torch.randn(1)], [torch.randn(1,3,224,224),torch.randn(1)], \
    # [torch.randn(1,3,224,224),torch.randn(1)]]

    ###################################33tres

    if(useGoogLeNet):
        print("************ Using GoogLeNet ************")
        net = GoogLeNet.MyGoogLeNet(train_obj_name,test_obj_name,usePreloadModel,withCuda,withLSTM)
    else:
        print("************ Using AlexNet ************")
        net = AlexNet.MyAlexNet(train_obj_name,test_obj_name,usePreloadModel,withCuda,withLSTM)

    if(isTraining):
        print("************ Training ************")
        ephocs, errors, times, accuracies = net.train()

        fig = plt.figure(1)
        firstSubPlot = plt.subplot(211)
        plt.plot(ephocs,errors,'r--')
        firstSubPlot.set_xlabel('Epochs')
        firstSubPlot.set_ylabel('Error')
        secondSubPlot = plt.subplot(212)
        plt.plot(ephocs,accuracies,'bs')
        secondSubPlot.set_xlabel('Epochs')
        secondSubPlot.set_ylabel('Train Acc')
        fig.savefig(fignames)
        # plt.show()
    else:
        print("************ Testing ************")
        test_loss, accuracy = net.test()
else:
    ###################################33tres
    ##Mock data for LSTM
    ######### Data should be a FloatTensor of dims [30*10,1,224*224] #########
    ######### Label should be a FloatTensor of dims [1] #########
    print("************ Preparing the data ************")

    # time_steps = 10*30
    # batch_size = 1
    # in_size = 224*224
    # classes_no = 8
    # train_loader = [[torch.randn(time_steps, batch_size, in_size),torch.FloatTensor(batch_size).random_(0, classes_no-1)], \
    # [torch.randn(time_steps, batch_size, in_size),torch.FloatTensor(batch_size).random_(0, classes_no-1)], \
    # [torch.randn(time_steps, batch_size, in_size),torch.FloatTensor(batch_size).random_(0, classes_no-1)], \
    # [torch.randn(time_steps, batch_size, in_size),torch.FloatTensor(batch_size).random_(0, classes_no-1)]]

    # test_loader = [[torch.randn(time_steps, batch_size, in_size),torch.FloatTensor(batch_size).random_(0, classes_no-1)], \
    # [torch.randn(time_steps, batch_size, in_size),torch.FloatTensor(batch_size).random_(0, classes_no-1)], \
    # [torch.randn(time_steps, batch_size, in_size),torch.FloatTensor(batch_size).random_(0, classes_no-1)]]
    ###################################

    print("************ Using Simple LSTM ************")
    net = LSTM.MyLSTM(train_obj_name,test_obj_name,usePreloadModel,withCuda,model)
    net = LSTM.MyLSTM(train_obj_name,test_obj_name,usePreloadModel,withCuda,model)
    if(isTraining):
        print("************ Training ************")
        ephocs, errors, times, accuracies = net.train()

        fig = plt.figure(1)
        firstSubPlot = plt.subplot(211)
        plt.plot(ephocs,errors,'r--')
        firstSubPlot.set_xlabel('Epochs')
        firstSubPlot.set_ylabel('Error')
        secondSubPlot = plt.subplot(212)
        plt.plot(ephocs,accuracies,'bs')
        secondSubPlot.set_xlabel('Epochs')
        secondSubPlot.set_ylabel('Accuracies')
        fig.savefig(fignames)
        # plt.show()
    else:
        print("************ Testing ************")
        test_loss, accuracy = net.test()
