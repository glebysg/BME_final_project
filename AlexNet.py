#!/usr/bin/env python3.6

'''AlexNet with PyTorch.'''
import math
import time
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from DataPooling import DataPool

#Hyper parameters
initial_learning_rate = 0.01
momentum = 0.9
epoch_count = 50
num_classes = 8
LSTM_hidden_layers = 32
stream_num = 20
# batch_size = 1

class MyAlexNet:
    def __init__(self,train_obj_name,test_obj_name,withPreloadModel,withCuda,withLSTM):
        self.model = AlexNet(withLSTM)
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
            self.model.load_state_dict(torch.load('model/AlexNetModelMomentum09only'))

        # self.train_loader = train_obj_name
        # self.test_loader = test_obj_name
        self.train_loader = DataPool(train_obj_name,stream_num)
        self.test_loader = DataPool(test_obj_name,stream_num)
        self.criterion = nn.CrossEntropyLoss()
        self.errors = []
        self.epochs = []
        self.times = []

        self.optimizer = optim.SGD(self.model.parameters(), initial_learning_rate, momentum)

    def train(self):
        def step_decay(epoch):
           drop = 0.5
           epochs_drop = 10.0
           lrate = initial_learning_rate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
           return lrate

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
            # for batch_idx, (data, target) in enumerate(self.test_loader):
                # get the next data, target (as tensors)

                # if (len(input.size()) == 1):
                ## NOTE: reshape of torch
                #     input = input.view(1, input.size()[0])

                # target = target.type(torch.LongTensor)
                # print(data.size())
                # print(target.size())

                # Convert variables if you are using cuda
                if(self.withCuda):
                    data, target = Variable(data.cuda()), Variable(target.cuda())
                else:
                    data, target = Variable(data), Variable(target)

                self.optimizer.zero_grad()
                # target = Variable(torch.randn(1).abs().cuda().long())
                # For the LSTM approach,retrieve the last element of the output sequence
                if(self.withLSTM):
                    output_seq, _ = self.model(data)
                    last_output = output_seq[-1]
                    loss = self.criterion(last_output, target)
                    pred = last_output.data.max(1, keepdim=True)[1] # get the index of the max log-probability

                # In the non-LSTM, use the whole output of the model
                else:
                    output = self.model(data)
                    # print(output)
                    # print(target)
                    # print("TARGEEEEEEETO")
                    # print(target.data[0])
                    loss = self.criterion(output, target)
                    # print(loss)
                    pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
                    # print(pred)
                
                # Update the prediction values
                correct += pred.eq(target.data.view_as(pred)).cpu().sum()

                # Perform a backward propagation
                loss.backward()
                self.optimizer.step()
                # print(cntr)
                cntr+=1
                if(cntr%1000==0):
                    print('Used Images: {}, Time taken: {}'.format(cntr,(time.time() - start_time)))
            # Append the error obtained from this particular epoch
            self.errors.append(100-correct/float(self.train_loader.total_images()))

        start_time = time.time()

        # Iterate for all the epochs
        for current_epoch in range(1, epoch_count):
            self.epochs.append(current_epoch)

            # new_lr = step_decay(current_epoch)
            # print ('Train Epoch: {}'.format(new_lr))
            # for param_group in self.optimizer.param_groups:
                # param_group['lr'] = new_lr

            innerTrain(current_epoch)

            print ('Train Epoch: {}, Time taken: {}'.format(current_epoch, (time.time() - start_time)))
            self.times.append(time.time() - start_time)

        # Save the trained model
        torch.save(self.model.state_dict(), 'model/AlexNetModelAdaptiveLR')

        return self.epochs,self.errors,self.times

    def test(self):
        # Set the model to test
        self.model.train(False)
        test_loss = 0
        correct = 0

        # Reset the data pooling
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

        # Compute the final loss and the accuracy
        test_loss /= self.test_loader.total_videos()
        accuracy = 100. * correct / self.train_loader.total_images()

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, \
        self.train_loader.total_images(), accuracy))

        return test_loss, accuracy

# AlexNet Module for pytorch
# https://github.com/pytorch/vision/blob/master/torchvision/models/alexnet.py
class AlexNet(nn.Module):
    def __init__(self, withLSTM):
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
        )

        # Final linear layer when not using LSTM
        self.linear = nn.Linear(4096, num_classes)

        # Final LSTM layer when using LSTM
        self.rnn = nn.LSTM(4096, num_classes, LSTM_hidden_layers)

        self.withLSTM = withLSTM

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        if(self.withLSTM):
            x = x.view(1, x.size(0), x.size(1))
            x = self.rnn(x)
        else:
            x = self.linear(x)
        return x
