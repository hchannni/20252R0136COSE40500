import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import importlib
import pickle

import torch
import torch.nn as nn
import math

from keras.datasets import mnist

import nn_layers_pt as nnl
importlib.reload(nnl)

class nn_mnist_classifier:
    def __init__(self, mmt_friction=0.9, lr=1e-2):
        self.conv_layer_1 = nnl.nn_convolutional_layer(f_height=3, f_width=3, input_size=28,
                                                       in_ch_size=1, out_ch_size=28)

        self.act_1 = nnl.nn_activation_layer()

        self.maxpool_layer_1 = nnl.nn_max_pooling_layer(pool_size=2, stride=2)

        self.fc1 = nnl.nn_fc_layer(input_size=28*13*13, output_size=128)
        self.act_2 = nnl.nn_activation_layer()

        self.fc2 = nnl.nn_fc_layer(input_size=128, output_size=10)

        self.sm1 = nnl.nn_softmax_layer()

        self.xent = nnl.nn_cross_entropy_layer()

        self.lr = lr
        self.mmt_friction = mmt_friction

    def forward(self, x, y):
        cv1_f = self.conv_layer_1.forward(x)

        ac1_f = self.act_1.forward(cv1_f)
        mp1_f = self.maxpool_layer_1.forward(ac1_f)

        fc1_f = self.fc1.forward(mp1_f)
        ac2_f = self.act_2.forward(fc1_f)

        fc2_f = self.fc2.forward(ac2_f)

        sm1_f = self.sm1.forward(fc2_f)

        cn_f = self.xent.forward(sm1_f, y)

        scores = sm1_f
        loss = cn_f

        return scores, loss
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    
    def step(self):

      self.conv_layer_1.step(self.lr, self.mmt_friction)
      self.fc1.step(self.lr, self.mmt_friction)
      self.fc2.step(self.lr, self.mmt_friction)

class MNISTClassifier_PT(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_layer_1 = nn.Conv2d(in_channels=1, out_channels=28, kernel_size=3)
        
        self.act_1 = nn.ReLU() 

        self.maxpool_layer_1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(in_features=28*13*13, out_features=128)
        self.act_2 = nn.ReLU()

        self.fc2 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):

        cv1_f = self.conv_layer_1(x)
        ac1_f = self.act_1(cv1_f)
        mp1_f = self.maxpool_layer_1(ac1_f)
        
        mp1_f = mp1_f.view(mp1_f.size(0), -1)
        
        fc1_f = self.fc1(mp1_f)
        ac2_f = self.act_2(fc1_f)
        
        out_logit = self.fc2(ac2_f)

        return out_logit


if __name__ == '__main__':
  
    torch.set_default_dtype(torch.float64)
    
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    X_train=np.expand_dims(X_train,axis=1)
    X_test=np.expand_dims(X_test,axis=1)

    print('Training data shape: ', X_train.shape)
    print('Training labels shape: ', y_train.shape)
    print('Test data shape: ', X_test.shape)
    print('Test labels shape: ', y_test.shape)

    n_train_sample = 50000
    n_val_sample = len(y_train) - n_train_sample

    X_train = X_train.astype('float64') / 255.0
    X_test = X_test.astype('float64') / 255.0
    
    X_s = np.split(X_train, [n_val_sample, ])
    X_val = X_s[0]
    X_train = X_s[1]

    y_s = np.split(y_train, [n_val_sample, ])
    y_val = y_s[0]
    y_train = y_s[1]
    
    trn_dataset=[]
    for d, l in zip(X_train, y_train):
        trn_dataset.append((d,l))
    
    val_dataset=[]
    for d, l in zip(X_val, y_val):
        val_dataset.append((d,l))
    
    test_dataset=[]
    for d, l in zip(X_test, y_test):
        test_dataset.append((d,l))

    lr = 0.01
    n_epoch = 2
    batch_size = 64
    val_batch = 100
    test_batch = 100

    friction = 0.9
    
    PYTORCH_BUILTIN = True

    if PYTORCH_BUILTIN:
        classifier = MNISTClassifier_PT()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(classifier.parameters(), lr=lr, momentum=friction)
    else:
        classifier = nn_mnist_classifier(mmt_friction=friction, lr=lr)

    numsteps = int(n_train_sample / batch_size)
    
    train_loader = torch.utils.data.DataLoader(
        trn_dataset, batch_size=batch_size, shuffle=True)
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=val_batch, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=test_batch, shuffle=True)

    do_validation = True

    for i in range(n_epoch):
        
        j = 0
        trn_accy = 0
        
        for trn_data in train_loader:
            X, y = trn_data
            
            X = torch.as_tensor(X)
            y = torch.as_tensor(y).long()
            
            if PYTORCH_BUILTIN:
                scores = classifier(X)
                loss = criterion(scores, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            else:
                scores, loss = classifier.forward(X, y)

                loss.backward()
                classifier.step()
            
            estim = torch.ravel(torch.argmax(scores, axis=1))
            trn_accy += torch.sum((estim == y).long()).item() / batch_size
            
            j+=1

            if (j + 1) % 50 == 0:
                print('loop count', j + 1)
                print('loss', loss.item())

                if (j + 1) % 200 == 0:
                    print('training accuracy:', trn_accy / 200 * 100, '%')
                    trn_accy = 0

                    if do_validation:
                        print('performing validation!')
                        X,y = next(iter(val_loader)) 

                        X = torch.as_tensor(X)
                        y = torch.as_tensor(y).long()
                        
                        with torch.no_grad():
                            
                            if PYTORCH_BUILTIN:
                                scores = classifier(X)
                            else:                                
                                scores, _ = classifier.forward(X, y)

                            estim = torch.ravel(torch.argmax(scores, axis=1))

                            val_accy = torch.sum((estim == y).long()).item()
                            print('validation accuracy:', val_accy, '%')

    test_batch = 100
    test_iter = int(y_test.shape[0] / test_batch)
    tot_accy = 0

    for test_data in test_loader:
        X, y = test_data

        X = torch.as_tensor(X)
        y = torch.as_tensor(y).long()

        with torch.no_grad():
            if PYTORCH_BUILTIN:
                scores = classifier(X)
            else:
                scores, _ = classifier.forward(X, y)
                
            estim = torch.ravel(torch.argmax(scores, axis=1))
            accy = torch.sum((estim == y).long()).item() / test_batch
            tot_accy += accy
        print('batch accuracy:', accy)

    print('total accuray', tot_accy / test_iter)

    plot_sample_prediction = True

    if plot_sample_prediction:
        num_plot = 10
        plt.figure(figsize=(12, 4))
        
        X_sample, y_sample = next(iter(test_loader))

        for i in range(num_plot):

            X = torch.as_tensor(X_sample[i:i+1])
            y = torch.as_tensor(y_sample[i]).long()

            if PYTORCH_BUILTIN:
                score = classifier(X)
            else:
                score, _ = classifier.forward(X, y)
            
            pred = torch.ravel(torch.argmax(score, axis=1))

            if y == pred:
              title_color = 'k'
            else:
              title_color = 'r'

            img = np.squeeze(X_sample[i])
            ax = plt.subplot(1, num_plot, i + 1)
            plt.imshow(img, cmap=plt.get_cmap('gray'))

            ax.set_title('GT:' + str(y.item()) + '\n Pred:' + str(int(pred)), color=title_color)

        plt.tight_layout()
        plt.show()