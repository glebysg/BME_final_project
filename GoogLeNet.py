#!/usr/bin/env python3.6

'''GoogLeNet with PyTorch.'''
import time
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

class MyGoogLeNet:
    def __init__(self,train_loader,test_loader,withPreloadModel,withCuda,withLSTM):
        self.model = GoogLeNet()
        self.withCuda = withCuda

        if(self.withCuda):
            self.model.cuda()

        if withPreloadModel:
            print("************ Loading Model ************")
            self.model.load_state_dict(torch.load('model/GoogLeNetModel'))

        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = nn.CrossEntropyLoss()       
        self.errors = []
        self.epochs = []
        self.times = []
        # self.totalCorrect = 0

        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.5)

    def train(self):
        def innerTrain(epoch):
            self.model.train(True)
            correct = 0
            for batch_idx, (data, target) in enumerate(self.train_loader):

                # if (len(input.size()) == 1):
                ## NOTE: reshape of torch
                #     input = input.view(1, input.size()[0])

                target = target.type(torch.LongTensor)

                if(self.withCuda):
                    data, target = Variable(data.cuda()), Variable(target.cuda())
                else:
                    data, target = Variable(data), Variable(target)

                self.optimizer.zero_grad()
                output = self.model(data)

                loss = self.criterion(output, target)
                pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
                correct += pred.eq(target.data.view_as(pred)).cpu().sum()
                loss.backward()
                self.optimizer.step()
            self.errors.append(100-correct/float(len(self.train_loader)))

        start_time = time.time()
        epoch_count = 10
        for current_epoch in range(1, epoch_count):
            self.epochs.append(current_epoch)
            innerTrain(current_epoch)
            print ('Train Epoch: {}, Time taken: {}'.format(current_epoch, (time.time() - start_time)))
            self.times.append(time.time() - start_time)

        torch.save(self.model.state_dict(), 'model/GoogLeNetModel')

        return self.epochs,self.errors,self.times

    def test(self):
        self.model.train(False)
        test_loss = 0
        correct = 0
        for batch_idx, (data, target) in enumerate(self.test_loader):
            target = target.type(torch.LongTensor)

            if(self.withCuda):
                data, target = Variable(data.cuda(), volatile=True), Variable(target.cuda(), volatile=True)
            else:
                data, target = Variable(data, volatile=True), Variable(target, volatile=True)

            output = self.model(data)
            loss = self.criterion(output, target)
            test_loss += loss.data.cpu().numpy()[0]
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        test_loss /= len(self.test_loader)
        accuracy = 100. * correct / (batch_idx+1)

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, \
        (batch_idx+1), accuracy))

        return test_loss, accuracy

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


class GoogLeNet(nn.Module):
    def __init__(self, num_classes = 8, hidden_layers=2):
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
        self.linear = nn.Linear(1024, num_classes)

        self.rnn = nn.LSTM(1024, num_classes, hidden_layers)

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
        out = out.view(out.size(0), -1)
        if(self.withLSTM):
            out = self.rnn(out)
        else:
            out = self.linear(out)
        return out
