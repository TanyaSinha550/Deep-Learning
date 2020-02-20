# AutoEncoders

# Importing the libraries
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

# Importing the dataset
movies = pd.read_csv('ml-1m/movies.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
users = pd.read_csv('ml-1m/users.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
ratings = pd.read_csv('ml-1m/ratings.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')

# Preparing the training set and the test set
training_set = pd.read_csv('ml-100k/u1.base', delimiter = '\t')
training_set = np.array(training_set, dtype = 'int')
test_set = pd.read_csv('ml-100k/u1.test', delimiter = '\t')
test_set = np.array(test_set, dtype = 'int')

# Getting the number of users and movies
nb_users = int(max(max(training_set[:,0]), max(test_set[:,0])))
nb_movies = int(max(max(training_set[:,1]), max(test_set[:,1])))

# Converting the data into an array with users in lines and movies in columns
def convert(data):
    new_data = []
    for id_users in range(1, nb_users + 1):
        id_movies = data[:,1][data[:,0] == id_users]
        id_ratings = data[:,2][data[:,0] == id_users]
        ratings = np.zeros(nb_movies)
        ratings[id_movies - 1] = id_ratings
        new_data.append(list(ratings))
    return new_data
training_set = convert(training_set)
test_set = convert(test_set)

# Converting the data into Torch tensors
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)


#Creating the architecture of the neural networks
''' Created a class to implement inheritance . Module class from nn is inherited.
 In init function , we have created the connection of stacked auto encoder which have 3 hidden layer and 
 input neurons = output neurons which is equal to nb_movies'''
class SAE(nn.Module):
    def __init__(self, ):
        super(SAE,self).__init__()
        self.fc1=nn.Linear(nb_movies,20) # connection between input layer and 1st hidden layer
        self.fc2=nn.Linear(20,10)   # connection between 1st hidden layer and 2nd hidden layer
        self.fc3=nn.Linear(10,20) ## connection betweeen 2nd hidden layer and 3rd hidden layer
        self.fc4=nn.Linear(20,nb_movies) #connection betwwn 3rd hidden layer and output layer
        self.activation=nn.Sigmoid()
    def forward(self,x):
        x=self.activation(self.fc1(x))
        x=self.activation(self.fc2(x))
        x=self.activation(self.fc3(x))
        x=self.fc4(x)
        return x
sae=SAE()
criterion=nn.MSELoss()
optimizer=optim.RMSprop(sae.parameters(),lr=0.01,weight_decay=0.5)
 #lr is learning rate and weight_decay is used to reduce the learning rate after some epochs



# Training the Stacked Auto Encoder
nb_epoch=200
for epoch in range(1,nb_epoch+1):
    train_loss=0
    s=0. # to count the users who have given atleast one rating
    for id_user in range(nb_users):
        input=Variable(training_set[id_user]).unsqueeze(0) #to create a batch we use Variable so to be in format taken by pytorch
        target=input.clone() #to get actual ratings of input
        if torch.sum(target.data >0)>0:
            output=sae(input) # predicted ratings
            target.require_grad=False # to apply gradient descent only on output , not on inputs
            output[target==0] = 0 # to make ratings 0 as 0 only and they are not updated
            loss=criterion(output,target)
            mean_corrector=nb_movies/float(torch.sum(target.data>0) + 1e-10) # dividing nb_movies by no of non zero artings and adding a small number to avoid being denominator zero
            loss.backward()
            train_loss+=np.sqrt(loss.data*mean_corrector)
            s+=1.
            optimizer.step()
        print('epoch' + str(epoch)+'loss'+str(train_loss/s))



# Testing the SAE
test_loss = 0
s = 0.
for id_user in range(nb_users):
    input = Variable(training_set[id_user]).unsqueeze(0)
    target = Variable(test_set[id_user])
    if torch.sum(target.data > 0) > 0:
        output = sae(input)
        target.require_grad = False
        output[(target == 0).unsqueeze(0)] = 0
        loss = criterion(output, target)
        mean_corrector = nb_movies/float(torch.sum(target.data > 0) + 1e-10)
        test_loss += np.sqrt(loss.data*mean_corrector)
        s += 1.
print('test loss: '+str(test_loss/s))


        
            
   







