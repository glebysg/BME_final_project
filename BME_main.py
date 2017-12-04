#!/usr/bin/env python3.6

import sys
import LSTM
import AlexNet
import GoogLeNet
import torch, torchvision
import matplotlib.pyplot as plt

import torch.nn as nn
from torch.autograd import Variable

useCNN = True
useGoogLeNet = True
usePreloadModel = True
isTraining = False
withCuda = True
withLSTM = True
net = []
train_obj_name = 'trainout'
test_obj_name = 'testout'

tensor = torch.randn(224,224,3)
tensor = tensor.view(1,-1,tensor.size(0),tensor.size(1))
print(tensor.size())
exit(0)

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
        ephocs, errors, times = net.train()

        fig = plt.figure(1)
        firstSubPlot = plt.subplot(211)
        plt.plot(ephocs,errors,'r--')
        firstSubPlot.set_xlabel('Epochs')
        firstSubPlot.set_ylabel('Error')   
        secondSubPlot = plt.subplot(212)
        plt.plot(ephocs,times,'bs')
        secondSubPlot.set_xlabel('Epochs')
        secondSubPlot.set_ylabel('Time')  
        plt.show()
    else:
        print("************ Testing ************")
        test_loss, accuracy = net.test()

else:
    ###################################33tres
    ##Mock data for LSTM
    ######### Data should be a FloatTensor of dims [30*10,1,224*224] #########
    ######### Label should be a FloatTensor of dims [1] #########    
    print("************ Preparing the data ************")

    time_steps = 10*30
    batch_size = 1
    in_size = 224*224
    classes_no = 8

    train_loader = [[torch.randn(time_steps, batch_size, in_size),torch.FloatTensor(batch_size).random_(0, classes_no-1)], \
    [torch.randn(time_steps, batch_size, in_size),torch.FloatTensor(batch_size).random_(0, classes_no-1)], \
    [torch.randn(time_steps, batch_size, in_size),torch.FloatTensor(batch_size).random_(0, classes_no-1)], \
    [torch.randn(time_steps, batch_size, in_size),torch.FloatTensor(batch_size).random_(0, classes_no-1)]]

    test_loader = [[torch.randn(time_steps, batch_size, in_size),torch.FloatTensor(batch_size).random_(0, classes_no-1)], \
    [torch.randn(time_steps, batch_size, in_size),torch.FloatTensor(batch_size).random_(0, classes_no-1)], \
    [torch.randn(time_steps, batch_size, in_size),torch.FloatTensor(batch_size).random_(0, classes_no-1)]]
    ###################################33tres

    print("************ Using Simple LSTM ************")
    net = LSTM.MyLSTM(train_loader,test_loader,usePreloadModel,withCuda)

    if(isTraining):
        print("************ Training ************")
        ephocs, errors, times = net.train()

        fig = plt.figure(1)
        firstSubPlot = plt.subplot(211)
        plt.plot(ephocs,errors,'r--')
        firstSubPlot.set_xlabel('Epochs')
        firstSubPlot.set_ylabel('Error')   
        secondSubPlot = plt.subplot(212)
        plt.plot(ephocs,times,'bs')
        secondSubPlot.set_xlabel('Epochs')
        secondSubPlot.set_ylabel('Time')  
        plt.show()
    else:
        print("************ Testing ************")
        test_loss, accuracy = net.test()
