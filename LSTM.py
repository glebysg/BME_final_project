#!/usr/bin/env python3.6

'''Simple LSTM with PyTorch.'''
import time
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

#Hyper parameters
learning_rate = 0.01
momentum = 0.5
epoch_count = 10
num_classes = 8
LSTM_hidden_layers = 2
# batch_size = 1

class MyLSTM:
    def __init__(self,train_loader,test_loader,withPreloadModel,withCuda):
        self.model = LSTM()
        self.withCuda = withCuda

        if(self.withCuda):
            self.model.cuda()

        # Load the trained model
        if withPreloadModel:
            print("************ Loading Model ************")
            self.model.load_state_dict(torch.load('model/LSTMModel'))

        self.train_loader = train_loader
        self.test_loader = test_loader
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
            # Reset the data pooling

            # loop for as many examples
            for batch_idx, (data, target) in enumerate(self.train_loader):

                # get the next data, target (as tensors)

                # if (len(input.size()) == 1):
                ## NOTE: reshape of torch
                #     input = input.view(1, input.size()[0])

                target = target.type(torch.LongTensor)

                # Convert variables if you are using cuda
                if(self.withCuda):
                    data, target = Variable(data.cuda()), Variable(target.cuda())
                else:
                    data, target = Variable(data), Variable(target)

                self.optimizer.zero_grad()

                # Retrieve the last element of the output sequence
                output_seq, _ = self.model(data)
                last_output = output_seq[-1]
                loss = self.criterion(last_output, target)
                pred = last_output.data.max(1, keepdim=True)[1] # get the index of the max log-probability

                # Update the prediction values
                correct += pred.eq(target.data.view_as(pred)).cpu().sum()

                # Perform a backward propagation
                loss.backward()
                self.optimizer.step()

            # Append the error obtained from this particular epoch
            self.errors.append(100-correct/float(len(self.train_loader)))

        start_time = time.time()

        # Iterate for all the epochs
        for current_epoch in range(1, epoch_count):
            self.epochs.append(current_epoch)
            innerTrain(current_epoch)
            print ('Train Epoch: {}, Time taken: {}'.format(current_epoch, (time.time() - start_time)))
            self.times.append(time.time() - start_time)

        # Save the trained model
        torch.save(self.model.state_dict(), 'model/LSTMModel')

        return self.epochs,self.errors,self.times

    def test(self):
        # Set the model to test
        self.model.train(False)
        test_loss = 0
        correct = 0

        # Reset the data pooling

        # loop for as many examples
        for batch_idx, (data, target) in enumerate(self.test_loader):
            # get the next data, target (as tensors)

            target = target.type(torch.LongTensor)

            # Convert variables if you are using cuda
            if(self.withCuda):
                data, target = Variable(data.cuda(), volatile=True), Variable(target.cuda(), volatile=True)
            else:
                data, target = Variable(data, volatile=True), Variable(target, volatile=True)

            # Retrieve the last element of the output sequence
            output_seq, _ = self.model(data)
            last_output = output_seq[-1]
            loss = self.criterion(last_output, target)
            pred = last_output.data.max(1, keepdim=True)[1] # get the index of the max log-probability

            # Update the prediction values
            test_loss += loss.data.cpu().numpy()[0]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        # Compute the final loss and the accuracy
        test_loss /= len(self.test_loader)
        accuracy = 100. * correct / (batch_idx+1)

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, \
        (batch_idx+1), accuracy))

        return test_loss, accuracy

# A simple neural network with only an LSTM layer
class LSTM(nn.Module):
    def __init__(self, in_size=224*224):
        super(LSTM, self).__init__()
        self.a1 = nn.LSTM(in_size, num_classes, LSTM_hidden_layers)

    def forward(self, x):
        out = self.a1(x)
        return out