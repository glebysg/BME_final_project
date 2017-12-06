#!/usr/bin/env python3.6

'''GoogLeNet with PyTorch.'''
import time
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from DataPooling import DataPool
from torch.autograd import Variable

#Hyper parameters
learning_rate = 0.01
momentum = 0.5
epoch_count = 50
num_classes = 8
LSTM_hidden_layers = 32
stream_num = 20
# batch_size = 1

class MyGoogLeNet:
    def __init__(self,train_obj_name,test_obj_name,withPreloadModel,withCuda,withLSTM):
        self.model = GoogLeNet(withLSTM)
        self.withCuda = withCuda
        self.withLSTM = withLSTM

        if(self.withLSTM):
            print("************ Using LSTM ************")
        else:
            print("************ Not Using LSTM ************")

        if(self.withCuda):
            self.model.cuda()

        # Load the trained model
        if withPreloadModel:
            print("************ Loading Model ************")
            self.model.load_state_dict(torch.load('model/GoogLeNetModel'))

        self.train_loader = DataPool(train_obj_name,stream_num)
        self.test_loader = DataPool(test_obj_name,stream_num)
        self.criterion = nn.CrossEntropyLoss()       
        self.errors = []
        self.epochs = []
        self.times = []

        self.optimizer = optim.SGD(self.model.parameters(), learning_rate, momentum)

    def train(self):
        def innerTrain(epoch):
            # Set the model to train
            self.model.train(True)
            correct = 0
            cntr = 0
            # Reset the data pooling
            # Loop for all the examples
            self.train_loader.restart()
            while(True):
                data,target = self.train_loader.nextImage()
                if(data is None):
                    break
            # loop for as many examples

            # for batch_idx, (data, target) in enumerate(self.train_loader):

                # get the next data, target (as tensors)

                # if (len(input.size()) == 1):
                ## NOTE: reshape of torch
                #     input = input.view(1, input.size()[0])

                # target = target.type(torch.LongTensor)

                # Convert variables if you are using cuda
                if(self.withCuda):
                    data, target = Variable(data.cuda()), Variable(target.cuda())
                else:
                    data, target = Variable(data), Variable(target)

                self.optimizer.zero_grad()

                # For the LSTM approach,retrieve the last element of the output sequence
                if(self.withLSTM):
                    output_seq, _ = self.model(data)
                    last_output = output_seq[-1]
                    loss = self.criterion(last_output, target)
                    pred = last_output.data.max(1, keepdim=True)[1] # get the index of the max log-probability

                # In the non-LSTM, use the whole output of the model
                else:
                    output = self.model(data)
                    loss = self.criterion(output, target)
                    pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability

                # Update the prediction values
                correct += pred.eq(target.data.view_as(pred)).cpu().sum()

                # Perform a backward propagation
                loss.backward()
                self.optimizer.step()
                cntr+=1
                if(cntr%1000==0):
                    print('Used Images: {}, Time taken: {}'.format(cntr,(time.time() - start_time)))

            # Append the error obtained from this particular epoch
            self.errors.append(100-correct/float(self.train_loader.total_videos()))

        start_time = time.time()

        # Iterate for all the epochs
        for current_epoch in range(1, epoch_count):
            self.epochs.append(current_epoch)
            innerTrain(current_epoch)
            print ('Train Epoch: {}, Time taken: {}'.format(current_epoch, (time.time() - start_time)))
            self.times.append(time.time() - start_time)

        # Save the trained model
        torch.save(self.model.state_dict(), 'model/GoogLeNetModel')

        return self.epochs,self.errors,self.times

    def test(self):
        # Set the model to test
        self.model.train(False)
        test_loss = 0
        correct = 0
        cntr = 0
        # Reset the data pooling
        # Loop for all the examples
        self.train_loader.restart()
        while(True):
            data,target = self.train_loader.nextImage()
            if(data is None):
                break
        # loop for as many examples
        # for batch_idx, (data, target) in enumerate(self.test_loader):
            # get the next data, target (as tensors)

            # target = target.type(torch.LongTensor)

            # Convert variables if you are using cuda
            if(self.withCuda):
                data, target = Variable(data.cuda(), volatile=True), Variable(target.cuda(), volatile=True)
            else:
                data, target = Variable(data, volatile=True), Variable(target, volatile=True)

            # For the LSTM approach,retrieve the last element of the output sequence
            if(self.withLSTM):
                output_seq, _ = self.model(data)
                last_output = output_seq[-1]
                loss = self.criterion(last_output, target)
                pred = last_output.data.max(1, keepdim=True)[1] # get the index of the max log-probability

            # In the non-LSTM, use the whole output of the model
            else:
                output = self.model(data)
                loss = self.criterion(output, target)
                pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability

            # Update the prediction values
            test_loss += loss.data.cpu().numpy()[0]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
            
            cntr+=1
            if(cntr%1000==0):
                print('Used Images: {}, Time taken: {}'.format(cntr,(time.time() - start_time)))
        # Compute the final loss and the accuracy
        test_loss /= len(self.test_loader)
        accuracy = 100. * correct / (batch_idx+1)

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, \
        (batch_idx+1), accuracy))

        return test_loss, accuracy

# Inception Module for the GoogLeNet
class Inception(nn.Module):
    def __init__(self, in_planes, n1x1, n3x3red, n3x3, n5x5red, n5x5, pool_planes):
        super(Inception, self).__init__()
        # 1x1 conv branch
        self.b1 = nn.Sequential(
            nn.Conv2d(in_planes, n1x1, kernel_size=1),
            nn.BatchNorm2d(n1x1),
            nn.ReLU(True),
        )

        # 1x1 conv -> 3x3 conv branch
        self.b2 = nn.Sequential(
            nn.Conv2d(in_planes, n3x3red, kernel_size=1),
            nn.BatchNorm2d(n3x3red),
            nn.ReLU(True),
            nn.Conv2d(n3x3red, n3x3, kernel_size=3, padding=1),
            nn.BatchNorm2d(n3x3),
            nn.ReLU(True),
        )

        # 1x1 conv -> 5x5 conv branch
        self.b3 = nn.Sequential(
            nn.Conv2d(in_planes, n5x5red, kernel_size=1),
            nn.BatchNorm2d(n5x5red),
            nn.ReLU(True),
            nn.Conv2d(n5x5red, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(n5x5),
            nn.ReLU(True),
            nn.Conv2d(n5x5, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(n5x5),
            nn.ReLU(True),
        )

        # 3x3 pool -> 1x1 conv branch
        self.b4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_planes, pool_planes, kernel_size=1),
            nn.BatchNorm2d(pool_planes),
            nn.ReLU(True),
        )

    def forward(self, x):
        y1 = self.b1(x)
        y2 = self.b2(x)
        y3 = self.b3(x)
        y4 = self.b4(x)
        return torch.cat([y1,y2,y3,y4], 1)

# GoogLeNet Module for pytorch
# https://github.com/kuangliu/pytorch-cifar/blob/master/models/googlenet.py
class GoogLeNet(nn.Module):
    def __init__(self,withLSTM):
        super(GoogLeNet, self).__init__()
        self.pre_layers = nn.Sequential(
            nn.Conv2d(3, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(True),
        )

        self.a3 = Inception(192,  64,  96, 128, 16, 32, 32)
        self.b3 = Inception(256, 128, 128, 192, 32, 96, 64)

        self.maxpool1 = nn.MaxPool2d(3, stride=2, padding=1)

        self.a4 = Inception(480, 192,  96, 208, 16,  48,  64)
        self.b4 = Inception(512, 160, 112, 224, 24,  64,  64)
        self.c4 = Inception(512, 128, 128, 256, 24,  64,  64)
        self.d4 = Inception(512, 112, 144, 288, 32,  64,  64)
        self.e4 = Inception(528, 256, 160, 320, 32, 128, 128)

        self.maxpool2 = nn.MaxPool2d(3, stride=2, padding=1)

        self.a5 = Inception(832, 256, 160, 320, 32, 128, 128)
        self.b5 = Inception(832, 384, 192, 384, 48, 128, 128)

        self.avgpool = nn.AvgPool2d(56, stride=1)

        # Final linear layer when not using LSTM
        self.linear = nn.Linear(1024, num_classes)

        # Final LSTM layer when using LSTM
        self.rnn = nn.LSTM(1024, num_classes, LSTM_hidden_layers)
        self.withLSTM = withLSTM

    def forward(self, x):
        out = self.pre_layers(x)
        out = self.a3(out)
        out = self.b3(out)
        out = self.maxpool1(out)
        out = self.a4(out)
        out = self.b4(out)
        out = self.c4(out)
        out = self.d4(out)
        out = self.e4(out)
        out = self.maxpool2(out)
        out = self.a5(out)
        out = self.b5(out)
        out = self.avgpool(out)
        if(self.withLSTM):
            out = out.view(1, out.size(0), -1)
            out = self.rnn(out)
        else:
            out = out.view(out.size(0), -1)
            out = self.linear(out)
        return out
