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

import RNN_model
from RNN_model import RNN_model

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

    x_train.append(line)
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

    x_test.append(line)
y_test = np.zeros((25000,))
y_test[0:12500] = 1

vocab_size += 1

no_hidden_units = 500
print('no_hidden_units: %4d'% no_hidden_units)

#model = BOW_model(no_hidden_units) # try 300 as well

model = torch.load('rnn_mod_04.model')

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
    
batch_size = 100
no_of_epochs = 10
L_Y_train = len(y_train)
L_Y_test = len(y_test)

print('batch_size: %4d'% batch_size)
print('num_epochs: %3d'% no_of_epochs)
#print('Exponential LR scheduler with gamma: %.2f'% gamma_in)

sys.stdout.flush()

test_accu = []

for epoch in range(no_of_epochs):
    
    # do testing loop

    model.eval()

    epoch_acc = 0.0
    epoch_loss = 0.0

    epoch_counter = 0
    
    sequence_length = (epoch+1)*50
    
    time1 = time.time()
    
    I_permutation = np.random.permutation(L_Y_test)

    for i in range(0, L_Y_test, batch_size):
        
        ## within the testing loop
        x_input2 = [x_test[j] for j in I_permutation[i:i+batch_size]]
        x_input = np.zeros((batch_size,sequence_length),dtype=np.int)
        for j in range(batch_size):
            x = np.asarray(x_input2[j])
            sl = x.shape[0]
            if(sl < sequence_length):
                x_input[j,0:sl] = x
            else:
                start_index = np.random.randint(sl-sequence_length+1)
                x_input[j,:] = x[start_index:(start_index+sequence_length)]
        x_input = glove_embeddings[x_input]
        y_input = y_test[I_permutation[i:i+batch_size]]
        
        data = Variable(torch.FloatTensor(x_input)).cuda()
        target = Variable(torch.FloatTensor(y_input)).cuda()
        
        optimizer.zero_grad()
        loss, pred = model(data,target,train=True)
        
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

    print("Testing:   epoch %3d" % epoch, "sequence length %.2f" % (sequence_length), "accuracy %.2f" % (epoch_acc*100.0), "loss %.4f" % epoch_loss, "time %.4f" % float(time_elapsed))
    sys.stdout.flush()

#torch.save(model,'rnn_default.model')
#data = [train_loss,train_accu,test_accu]
#data = np.asarray(data)
#np.save('data_default.npy',data)
