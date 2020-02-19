#importing the libraries
'''torch.nn module is used to implement neural networks , torch.optim for optimizer , 
torch.autograd for stocastic gradient descent '''
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

# importing the dataset
'''separator is double semi-colon , header is not there in file , engine specify just the language
, encoding is done since movie names contains special characters so default encoding is not used'''

movies=pd.read_csv('ml-1m/movies.dat',sep='::',header=None,engine='python',encoding='latin-1')
users=pd.read_csv('ml-1m/users.dat',sep='::',header=None,engine='python',encoding='latin-1')
ratings=pd.read_csv('ml-1m/ratings.dat',sep='::',header=None,engine='python',encoding='latin-1')


# Preparing the training_set and train_set
''' the separartor is tab so use delimiter instead of sep '''

training_set=pd.read_csv('ml-100k/u1.base',delimiter='\t')
training_set=np.array(training_set,dtype='int')
test_set = pd.read_csv('ml-100k/u1.test', delimiter = '\t')
test_set = np.array(test_set, dtype = 'int')

#getting the number of users and movies
nb_users=int(max(max(training_set[:,0]),max(test_set[:,0])))
nb_movies=int(max(max(training_set[:,1]),max(test_set[:,1])))

#converting the data into an array with users in line and movies in columns
def convert(data):
    new_data=[]
    for id_users in range(1,nb_users+1):
        id_movies=data[:,1][data[:,0]==id_users]
        id_ratings=data[:,2][data[:,0]==id_users]
        ratings=np.zeros(nb_movies)
        ratings[id_movies-1]=id_ratings
        new_data.append(list(ratings))
    return new_data
training_set=convert(training_set)
test_set=convert(test_set)


#converting the data into torch tensors 
# tensors are pytorch arrays , it is amulti-dimensional matrix
training_set=torch.FloatTensor(training_set)
test_set=torch.FloatTensor(test_set)

# converting the ratings into binary format 0 and 1
training_set[training_set==0]=-1
training_set[training_set==1]=0
training_set[training_set==2]=0
training_set[training_set>=3]=1
test_set[test_set==0]=-1
test_set[test_set==1]=0
test_set[test_set==2]=0
test_set[test_set>=3]=1

# Creating the architecture of the neural networks
''' Creating a class with 3 functions
1) init function is used to initialize weights using radn having normal distribution,
   bias of the hidden node and it accepts two dimension i.e one batch size and second hidden nodes so we use (1,nh)
   bias of the visible node
2) sample_h is used just to apply sigmoid activation function to (w*nv+a) .Here x is visible node.
   sample_h is about sampling the hidden node according to the probability of the visible node
   we do this sampling because to approximate the log likelihood gradient which is done by Gibbs Sampling
   so first we calculate the probability of hidden node and then apply activation function
3) train is used to apply contrastive divergence to approximate log likelihood gradient
   v0 is visible nodes at first iteration
   vk is visible nodes at k iteration
   ph0 is probability of hidden nodes at first iteration
   phk is probability of hidden nodes at k iteration
'''
class RBM():
    def __init__(self,nv,nh):
        self.W=torch.randn(nh,nv)
        self.a=torch.randn(1,nh)
        self.b=torch.randn(1,nv)
    def sample_h(self,x):
        wx=torch.mm(x,self.W.t())
        activation=wx+self.a.expand_as(wx)
        p_h_given_v=torch.sigmoid(activation)
        return p_h_given_v,torch.bernoulli(p_h_given_v)
    def sample_v(self, y):
        wy = torch.mm(y, self.W)
        activation = wy + self.b.expand_as(wy)
        p_v_given_h = torch.sigmoid(activation)
        return p_v_given_h, torch.bernoulli(p_v_given_h)
    def train(self, v0, vk, ph0, phk):
        self.W += (torch.mm(v0.t(), ph0) - torch.mm(vk.t(), phk)).t()
        self.b += torch.sum((v0 - vk), 0)
        self.a += torch.sum((ph0 - phk), 0)
        
      
nv = len(training_set[0])
nh = 100
batch_size = 100
rbm = RBM(nv, nh)


#Training the RBM 
''' In the 3rd for loop , k is used to for k times contrastive divergence  '''
nb_epoch=10
for epoch in range(1,nb_epoch+1):
    train_loss=0
    s=0.
    for id_user in range(0,nb_users-batch_size,batch_size):
        vk=training_set[id_user:id_user+batch_size]
        v0 = training_set[id_user:id_user+batch_size]
        ph0,_ = rbm.sample_h(v0)
        for k in range(10):
            _,hk = rbm.sample_h(vk)
            _,vk = rbm.sample_v(hk)
            vk[v0<0] = v0[v0<0]
        phk,_ = rbm.sample_h(vk)
        rbm.train(v0, vk, ph0, phk)
        train_loss += torch.mean(torch.abs(v0[v0>=0] - vk[v0>=0]))
        s += 1.
    print('epoch: '+str(epoch)+' loss: '+str(train_loss/s))   
            
    

# Testing the RBM
test_loss = 0
s = 0.
for id_user in range(nb_users):
    v = training_set[id_user:id_user+1]
    vt = test_set[id_user:id_user+1]
    if len(vt[vt>=0]) > 0:
        _,h = rbm.sample_h(v)
        _,v = rbm.sample_v(h)
        test_loss += torch.mean(torch.abs(vt[vt>=0] - v[vt>=0]))
        s += 1.
print('test loss: '+str(test_loss/s))
            


 
    






