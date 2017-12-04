#!/usr/bin/env python3.6

import torch, cv2, torchvision, numpy as np, time
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
from torch.autograd import Variable

from time import gmtime, strftime

# Hyper Parameters 
hidden_size = 500
num_classes = 9
num_epochs = 18
batch_size = 50
learning_rate = 0.01

# https://github.com/eladhoffer/convNet.pytorch/blob/master/models/alexnet.py
class AlexNetOWT_BN(nn.Module):

    def __init__(self, num_classes = 9):
        super(AlexNetOWT_BN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2, bias=False),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 192, kernel_size=5, padding=2, bias=False),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(192),
            nn.Conv2d(192, 384, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(384),
            nn.Conv2d(384, 256, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256)
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 * 6 * 6, 4096, bias=False),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096, bias=False),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes)
        )

        self.regime = [
            {'epoch': 0, 'optimizer': 'SGD', 'lr': 1e-2, 'weight_decay': 5e-4, \
            'momentum': 0.9},
            {'epoch': 10, 'lr': 5e-3},
            {'epoch': 15, 'lr': 1e-3, 'weight_decay': 0},
            {'epoch': 20, 'lr': 5e-4},
            {'epoch': 25, 'lr': 1e-4}
        ]
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        self.input_transform = {
            'train': transforms.Compose([
                transforms.ToPILImage(),
                transforms.Scale(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ]),
            'eval': transforms.Compose([
                transforms.ToPILImage(),
                transforms.Scale(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize
            ])
        }

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 256 * 6 * 6)
        x = self.classifier(x)
        return x

# class customDataset_loader(Dataset):
#     TARGET_DIR = '/home/isat-deep/ros_ws/src/ie590project_ubuntu1404/data/FMD/image'

#     def __init__(self, split = 'image', transform = None, data_dir = None):
#         # load all images
#         self.TARGET_DIR = self.TARGET_DIR if data_dir is None else data_dir

#         class_list = [dir_namae for dir_namae in os.listdir(self.TARGET_DIR) \
#             if os.path.isdir(os.path.join(self.TARGET_DIR, dir_namae))]


#         # count number of image
#         N_all_img = 0
#         for ct in range(0, len(class_list)):
#             class_dir = self.TARGET_DIR + '/' + class_list[ct]
#             # print class_dir
#             class_files = glob.glob(self.TARGET_DIR + '/' + class_list[ct] + \
#                 '/*.jpg')

#             N_all_img += len(class_files)


#         img_array = np.zeros((N_all_img, 384, 512, 3)).astype(np.uint8)
#         class_label = []

#         ctx = 0
#         for ct in range(0, len(class_list)):
#             class_dir = self.TARGET_DIR + '/' + class_list[ct]
#             # print class_dir
#             class_files = glob.glob(self.TARGET_DIR + '/' + class_list[ct] + \
#                 '/*.jpg')

#             for fn in class_files:
#                 img_array[ctx] = cv2.imread(fn)
#                 class_label += [ct]
#                 ctx += 1

#         self.transform = transform
#         self.images = img_array  #torch.from_numpy(images)
#         self.labels = class_label  #torch.from_numpy(labels)
#         self.classes = class_list

#     def __getitem__(self, index):
#         img   = self.images[index]
#         label = self.labels[index]

#         #img = PIL.Image.fromarray(img)
#         if self.transform is not None:
#             img = self.transform(img)

#         return img, label

#     def __len__(self):
#         return len(self.images)

# class customDataset_loader_train1000(Dataset):
#     TARGET_DIR = '/home/isat-deep/ros_ws/src/ie590project_ubuntu1404/data/images_training_1000'

#     def __init__(self, split = 'image', transform = None, data_dir = None):
#         # load all images
#         self.TARGET_DIR = self.TARGET_DIR if data_dir is None else data_dir

#         class_list = [dir_namae for dir_namae in os.listdir(self.TARGET_DIR) \
#             if os.path.isdir(os.path.join(self.TARGET_DIR, dir_namae))]

#         print (len(class_list))
#         # count number of image
#         N_all_img = 0
#         for ct in range(0, len(class_list)):
#             class_dir = self.TARGET_DIR + '/' + class_list[ct]
#             # print class_dir
#             class_files = glob.glob(self.TARGET_DIR + '/' + class_list[ct] + \
#                 '/*.jpg')

#             N_all_img += len(class_files)

#         img_array = np.zeros((N_all_img, 256, 256, 3)).astype(np.uint8)
#         class_label = []

#         ctx = 0
#         for ct in range(0, len(class_list)):
#             class_dir = self.TARGET_DIR + '/' + class_list[ct]
#             # print class_dir
#             class_files = glob.glob(self.TARGET_DIR + '/' + class_list[ct] + \
#                 '/*.jpg')

#             for fn in class_files:
#                 img_in = cv2.imread(fn)
#                 y, x, z = img_in.shape

#                 dx = 256.0/x
#                 dy = 256.0/y

#                 img_array[ctx] = cv2.resize(img_in, None, fx = dx, fy = dy, interpolation = cv2.INTER_LINEAR)
#                 class_label += [ct]
#                 ctx += 1

#         self.transform = transform
#         self.images = img_array  #torch.from_numpy(images)
#         self.labels = class_label  #torch.from_numpy(labels)
#         self.classes = class_list

#     def __getitem__(self, index):
#         img   = self.images[index]
#         label = self.labels[index]

#         #img = PIL.Image.fromarray(img)
#         if self.transform is not None:
#             img = self.transform(img)

#         return img, label

#     def __len__(self):
#         return len(self.images)

def train_model(model, criterion, optimizer, scheduler, train_loader, test_loader, batch_size, num_epochs=25):
    since = time.time()

    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0.0
            num_sample = 0.0

            # Iterate over data.
            if phase == 'train':
                dataloaders = train_loader
            elif phase == 'val':
                dataloaders = test_loader
            else: 
                print ('wait ... what? we do not have: ', phase)

            for data in dataloaders:
                # get the inputs
                inputs, labels = data

                inputs = Variable(inputs.cuda())
                labels = Variable(labels.cuda())

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.data[0]
                running_corrects += torch.sum(preds == labels.data)
                num_sample += len(preds)

            # print 'num_sample = ', num_sample  

            epoch_loss = running_loss / num_sample
            epoch_acc = running_corrects / num_sample

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

########################################## 
########################################## From scratch
########################################## 

# alex_net = AlexNetOWT_BN()
# alexnet = models.alexnet(pretrained=True)
# alexnet.cuda()
# print ('Load images')
# dataset = customDataset_loader_train1000('t1000', transform = alex_net.input_transform['train'])

# # Loading data
# print ('Train/Test data loader')
# train_loader = torch.utils.data.DataLoader(dataset = dataset, 
#                                            batch_size = batch_size, 
#                                            shuffle = True)

# test_loader = torch.utils.data.DataLoader(dataset = dataset, 
#                                           batch_size = batch_size, 
#                                           shuffle = False)

# alex_net = AlexNetOWT_BN(num_classes = len(dataset.classes))
# alex_net.cuda()   

# # Loss and Optimizer
# print ('Training')
# criterion = nn.CrossEntropyLoss()  
# optimizer = torch.optim.Adam(alex_net.parameters(), lr = learning_rate)  

# # Train the Model
# for epoch in range(num_epochs):
#     for i, (images, labels) in enumerate(train_loader):  
#         label = labels
#         # Convert torch tensor to Variable
#         images = Variable(images.cuda())
#         labels = Variable(labels.cuda())
        
#         # Forward + Backward + Optimize
#         optimizer.zero_grad()  # zero the gradient buffer
#         outputs = alex_net(images)
#         loss = criterion(outputs, labels)
#         loss.backward()

#         _, predicted = torch.max(outputs.data, 1)
#         correct = (predicted.cpu() == label).sum()

#         optimizer.step()
        
#         if (i+1) % 10 == 0:
#             print ('Epoch [%d/%d], Loss: %.4f, Accuracy: %.4f' 
#                    %(epoch+1, num_epochs, loss.data[0], float(correct)/batch_size))

# # Test the Model
# correct = 0
# total = 0
# for images, labels in test_loader:
#     images = Variable(images.cuda())
#     outputs = alexnet(images)
#     _, predicted = torch.max(outputs.data, 1)
#     total += labels.size(0)
#     print (predicted.cpu(), labels )
#     correct += (predicted.cpu() == labels).sum()

# print('Accuracy of the network on the test images: %d %%' % (100 * float(correct)/total))

# # Save the Model
# torch.save(alex_net.state_dict(), 'model.pkl')

# print (strftime("%Y-%m-%d %H:%M:%S", gmtime()))