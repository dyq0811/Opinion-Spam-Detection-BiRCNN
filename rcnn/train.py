"""
Train the network. Adapted from the network written by 
Prakash Pandey.

https://github.com/prakashpandey9/Text-Classification-Pytorch
"""
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from net import Net
import load_data

def train_net(net, data, label, nepochs, batch_size, eval_single_output_fn = None):
    '''
    Trains the given RCNN using criterion to calculate the loss and optimizer
    to adjust the weights. 
    Prints out the training loss, training accuracy,
    and test accuracy every epoch.
    '''
    print("batch_size:", batch_size)
    print("nepochs:", nepochs)
    for epoch in range(1,nepochs+1):
        # TrainX is a list of batches for training. Each batch if a vector of size 
        # (batch_size, num_seq, embedding_length). 
        # TrainY is a list of labels. 
        # TestX and testY are of the same format for testing.
        trainX, trainY, testX, testY = load_data.split_data(data, label, batch_size)
        train_loss, train_acc = train_pattern(net, trainX, trainY, epoch)
        test_loss, test_acc = eval_model(net, testX, testY)
        print("epoch  %s: train loss %2.2f train accuracy %2.2f\n test loss %2.2f test accuracy %2.2f" % (epoch, train_loss, train_acc, test_loss, test_acc))
            
def train_pattern(net, training_data, training_label, epoch):
    '''
    Trains the given network.
    
    Params: 
    Training data: batches of review. Each batch is represented as 
    a tensor of size (batch_size, num_sequences, embedding_length).
    Training label: the expected categorization of the review.
    Epoch: the current epoch.
    '''
    total_epoch_loss = 0
    total_epoch_acc = 0
    steps = 0
    # Set the network in train mode and clear any previous gradients
    net.train()
    net.zero_grad()
    for idx, batch in enumerate(training_data):
        text = batch
        target = training_label[idx]
        target = torch.autograd.Variable(target).long()
        # Network categorization of the current batch.
        # Represented with a tensor of size (batch_size, 2).
        output = net(text)
        loss = criterion(output, target)      
        num_corrects = (torch.max(output, 1)[1].view(target.size()).data == target.data).float().sum()
        acc = 100.0 * num_corrects/len(batch)
        loss.backward()
        optimizer.step()
        steps += 1
        
#        if steps % 20 == 0:
#            print (f'Epoch: {epoch+1}, Idx: {idx+1}, Training Loss: {loss.item():.4f}, Training Accuracy: {acc.item(): .2f}%')
        
        total_epoch_loss += loss.item()
        total_epoch_acc += acc.item()
        
    return total_epoch_loss/len(training_data), total_epoch_acc/len(training_data)


def eval_model(net, testing_data, testing_label):
    """Evaluate the model."""
    total_epoch_loss = 0
    total_epoch_acc = 0
    # Set the network in eval mode
    net.eval()
    with torch.no_grad():
        for idx, batch in enumerate(testing_data):
            text = batch
            target = testing_label[idx]
            target = torch.autograd.Variable(target).long()
            
            output = net(text)
            loss = criterion(output, target)
            
            num_corrects = (torch.max(output, 1)[1].view(target.size()).data == target.data).sum()
            acc = 100.0 * num_corrects/len(batch)
            total_epoch_loss += loss.item()
            total_epoch_acc += acc.item()

    return total_epoch_loss/len(testing_data), total_epoch_acc/len(testing_data)

def eval_single_pattern(net, data, label):
    """Evaluate the model."""
    # Set the network in eval mode
    net.eval()
    wrong_labels = []
    with torch.no_grad():
        for idx, batch in enumerate(data):
            text = batch
            target = label[idx]
            target = torch.autograd.Variable(target).long()
            output = net(text,1)
            
            if torch.max(output, 1)[1].view(target.size()).data != target.data:
                wrong_labels.append(idx)

    return wrong_labels

def get_mistakes(net, data, label):
    # Mistakes that the trained model make
    data_tensor = load_data.transform(data, 1)
    label_tensor = torch.tensor([label[i] for i in range(0, len(label))])
    mistakes = eval_single_pattern(net, data_tensor, label_tensor)
    wrong_labels = []
    wrong_length = []
    for m in mistakes:
        wrong_labels.append(label[m])
        wrong_length.append(len(data[m]))
    print(wrong_labels)
    print(wrong_length)


# Initialize the constants
learning_rate = .01
batch_size = 16
nepochs = 10
hidden_size = 128

# Construct the net
net = Net(batch_size, hidden_size)
optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate) # stochastic gradient descent
criterion = F.cross_entropy #log-likelihood loss function

# Load the data
data, label = load_data.load()

# Load only positive data
pos_data, pos_label = load_data.load_pos()

# Load only negative data
neg_data, neg_label = load_data.load_neg()

# Train the network
train_net(net, data, label, nepochs, batch_size)

get_mistakes(net, data, label)
