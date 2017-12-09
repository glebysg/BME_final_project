#!/usr/bin/env python3.6

'''Simple LSTM with PyTorch.'''
import math
import time
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from DataPooling import DataPool
from torch.autograd import Variable

#Hyper parameters
initial_learning_rate = 0.1
momentum = 0.9
epoch_count = 50
num_classes = 8
LSTM_hidden_layers = 64
stream_num = 20
# batch_size = 1

class MyLSTM:
    def __init__(self,train_obj_name,test_obj_name,withPreloadModel,withCuda,modelName):
        self.model = LSTM()
        self.withCuda = withCuda
        self.model_name = 'model/'+modelName

        if(self.withCuda):
            self.model.cuda()

        # Load the trained model
        if withPreloadModel:
            print("************ Loading Model ************")
            self.model.load_state_dict(torch.load(self.model_name))

        self.train_loader = DataPool(train_obj_name,stream_num,'LSTM')
        self.test_loader = DataPool(test_obj_name,stream_num,'LSTM')
        self.criterion = nn.NLLLoss()
        self.errors = []
        self.epochs = []
        self.times = []
        self.accuracies = []

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
            cntr = 0
            correct = 0
            # Reset the data pooling

            # loop for as many examples
            self.train_loader.restart()
            while(True):
                data,target = self.train_loader.nextImage()
                if(data is None):
                    break
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
                cntr+=1
                if(cntr%100==0):
                    print(loss)
                    print('Used Images: {}, Time taken: {}, Loss: {}, Train Acc: {}'.format(cntr*20,(time.time() - \
                            start_time),loss.data[0],100.*(correct/cntr)))
            # Append the error obtained from this particular epoch
            self.errors.append(100-correct/float(self.train_loader.total_images()))
            self.accuracies.append(100.*(correct/self.train_loader.total_images()))

        start_time = time.time()

        for current_epoch in range(1, epoch_count):
            self.epochs.append(current_epoch)
            new_lr = step_decay(current_epoch)
            print ('Train Epoch learning rate: {}'.format(new_lr))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = new_lr

            innerTrain(current_epoch)

            print ('Train Epoch: {}, Time taken: {}'.format(current_epoch, (time.time() - start_time)))
            self.times.append(time.time() - start_time)

        # Save the trained model
        torch.save(self.model.state_dict(), self.model_name)

        return self.epochs,self.errors,self.times,self.accuracies

    def test(self):
        # Set the model to test
        self.model.train(False)
        test_loss = 0
        correct = 0
        cntr = 0
        # Reset the data pooling
        start_time = time.time()
        # Loop for all the examples
        self.test_loader.restart()
        while(True):
            data,target = self.test_loader.nextImage()
            if(data is None):
                break

        # Reset the data pooling
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
