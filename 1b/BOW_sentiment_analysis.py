import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch.distributed as dist

import time
import os
import sys
import io

from BOW_model import BOW_model

glove_embeddings = np.load('../preprocessed_data/glove_embeddings.npy')
vocab_size = 100000
print('vocab_size: %5d'% vocab_size)

x_train = []
with io.open('../preprocessed_data/imdb_train_glove.txt','r',encoding='utf-8') as f:
    lines = f.readlines()
for line in lines:
    line = line.strip()
    line = line.split(' ')
    line = np.asarray(line,dtype=np.int)

    line[line>vocab_size] = 0
    line = line[line!=0]

    line = np.mean(glove_embeddings[line],axis=0)

    x_train.append(line)
x_train = np.asarray(x_train)
x_train = x_train[0:25000]
y_train = np.zeros((25000,))
y_train[0:12500] = 1

x_test = []
with io.open('../preprocessed_data/imdb_test_glove.txt','r',encoding='utf-8') as f:
    lines = f.readlines()
for line in lines:
    line = line.strip()
    line = line.split(' ')
    line = np.asarray(line,dtype=np.int)

    line[line>vocab_size] = 0
    line = line[line!=0]
    
    line = np.mean(glove_embeddings[line],axis=0)

    x_test.append(line)
x_test = np.asarray(x_test)
y_test = np.zeros((25000,))
y_test[0:12500] = 1

vocab_size += 1

no_hidden_units = 100
print('no_hidden_units: %4d'% no_hidden_units)

model = BOW_model(no_hidden_units) # try 300 as well

model.cuda()

#opt = 'sgd'
#LR = 0.01
#gamma_in = 0.9
opt = 'adam'
LR = 0.001
if(opt=='adam'):
    optimizer = optim.Adam(model.parameters(), lr=LR)
elif(opt=='sgd'):
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9)
    
print('optimizer: %s'% opt)
print('LR: %.6f'% LR)

#scheduler = optim.lr_scheduler.MultiStepLR(optimizer,milestones = [40,70],gamma=0.1)
#scheduler = optim.lr_scheduler.ExponentialLR(optimizer,gamma=gamma_in)
    
batch_size = 200
no_of_epochs = 50
L_Y_train = len(y_train)
L_Y_test = len(y_test)

print('batch_size: %4d'% batch_size)
print('num_epochs: %3d'% no_of_epochs)
#print('Exponential LR scheduler with gamma: %.2f'% gamma_in)

sys.stdout.flush()

model.train()

train_loss = []
train_accu = []
test_accu = []

for epoch in range(no_of_epochs):

    # training
    model.train()

    epoch_acc = 0.0
    epoch_loss = 0.0

    epoch_counter = 0

    time1 = time.time()
    
    #scheduler.step()
    
    #for param_group in optimizer.param_groups:
    #        print("Epoch # %3d,  Learning Rate %10.6f" % ( epoch, param_group['lr'] ) )
    
    I_permutation = np.random.permutation(L_Y_train)

    for i in range(0, L_Y_train, batch_size):
        
        ## within the training loop
        x_input = x_train[I_permutation[i:i+batch_size]]
        y_input = y_train[I_permutation[i:i+batch_size]]

        data = Variable(torch.FloatTensor(x_input)).cuda()
        target = Variable(torch.FloatTensor(y_input)).cuda()

        optimizer.zero_grad()
        loss, pred = model(data,target)
        loss.backward()

        optimizer.step()   # update weights
        
        prediction = pred >= 0.0
        truth = target >= 0.5
        acc = prediction.eq(truth).sum().cpu().data.numpy()
        epoch_acc += acc
        epoch_loss += loss.data.item()
        epoch_counter += batch_size

    epoch_acc /= epoch_counter
    epoch_loss /= (epoch_counter/batch_size)

    train_loss.append(epoch_loss)
    train_accu.append(epoch_acc)

    print("Training: epoch %3d" % epoch, "accuracy %.2f" % (epoch_acc*100.0), "loss %.4f" % epoch_loss, "time %.4f" % float(time.time()-time1))

    # ## test
    model.eval()

    epoch_acc = 0.0
    epoch_loss = 0.0

    epoch_counter = 0

    time1 = time.time()
    
    I_permutation = np.random.permutation(L_Y_test)

    for i in range(0, L_Y_test, batch_size):
        
        ## within the training loop
        x_input = x_test[I_permutation[i:i+batch_size]]
        y_input = y_test[I_permutation[i:i+batch_size]]

        data = Variable(torch.FloatTensor(x_input)).cuda()
        target = Variable(torch.FloatTensor(y_input)).cuda()

        with torch.no_grad():
            loss, pred = model(data,target)
        
        prediction = pred >= 0.0
        truth = target >= 0.5
        acc = prediction.eq(truth).sum().cpu().data.numpy()
        epoch_acc += acc
        epoch_loss += loss.data.item()
        epoch_counter += batch_size

    epoch_acc /= epoch_counter
    epoch_loss /= (epoch_counter/batch_size)

    test_accu.append(epoch_acc)

    time2 = time.time()
    time_elapsed = time2 - time1

    print("Testing:           ", "accuracy %.2f" % (epoch_acc*100.0), "loss %.4f" % epoch_loss)
    sys.stdout.flush()

torch.save(model,'BOW_mod_09.model')
data = [train_loss,train_accu,test_accu]
data = np.asarray(data)
np.save('data_mod_09.npy',data)
