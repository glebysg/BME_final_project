#!/usr/bin/env python3.6

'''AlexNet with PyTorch.'''
import time
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

class MyAlexNet:
    def __init__(self,train_loader,test_loader,withPreloadModel,withCuda):
        self.model = AlexNet()
        self.withCuda = withCuda

        if(self.withCuda):
            self.model.cuda()

        if withPreloadModel:
            print("************ Loading Model ************")
            self.model.load_state_dict(torch.load('model/AlexNetModel'))

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
            self.model.train()
            correct = 0
            for batch_idx, (data, target) in enumerate(self.train_loader):

                # if (len(input.size()) == 1):
                #     input = input.view(1, input.size()[0])
                
                target = target.type(torch.LongTensor)

                if(self.withCuda):
                    data, target = Variable(data.cuda(),), Variable(target.cuda())
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

        torch.save(self.model.state_dict(), 'model/AlexNetModel')

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

class AlexNet(nn.Module):

    def __init__(self, num_classes = 8):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x
