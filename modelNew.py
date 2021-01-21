import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import DataLoader
from data import getXY

class Net(nn.Module):
    def __init__(self):
        super().__init__()

        # #400
        # self.c1=nn.Conv2d(5,400,kernel_size=(4,4))
        # self.c2=nn.Conv2d(400,400,kernel_size=(1,1))
        # self.c3=nn.Conv2d(400,400,kernel_size=(1,1))

        # #800
        # self.c4=nn.Conv2d(400,800,kernel_size=(4,4))
        # self.c5=nn.Conv2d(800,800,kernel_size=(1,1))
        # self.c6=nn.Conv2d(800,800,kernel_size=(1,1))

        # #1600
        # self.c7=nn.Conv2d(800,1600,kernel_size=(2,2))

        # self.l1=nn.Linear(1600,1)
       	self.convs=nn.Sequential(
       		nn.Conv2d(5,5,16,padding=16),nn.ReLU(),
       		
          nn.Conv2d(5,16,4),nn.ReLU(),
       		nn.Conv2d(16,16,2),nn.ReLU(),
       		
          nn.Conv2d(16,32,4),nn.ReLU(),
       		nn.Conv2d(32,32,2),nn.ReLU(),
       		
          nn.Conv2d(32,64,4),nn.ReLU(),
       		nn.Conv2d(64,64,2),nn.ReLU(),
       		
          nn.Conv2d(64,128,4),nn.ReLU(),
       		nn.Conv2d(128,128,2),nn.ReLU(),
          
          nn.Conv2d(128,256,4),nn.ReLU(),
          nn.Conv2d(256,256,2),nn.ReLU(),

          nn.Conv2d(256,256,5),nn.ReLU(),

          # nn.Conv2d(512,1024,2),nn.ReLU(),
          # nn.Conv2d(1024,1024,1),nn.ReLU(),
       		)
        self.lins=nn.Sequential(
          nn.Linear(256,256),nn.ReLU(),
          nn.Linear(256,128),nn.ReLU(),
          
          nn.Linear(128,128),nn.ReLU(),
          nn.Linear(128,64),nn.ReLU(),

          nn.Linear(64,64),nn.ReLU(),
          nn.Linear(64,32),nn.ReLU(),

          nn.Linear(32,32),nn.ReLU(),
          nn.Linear(32,16),nn.ReLU(),

          nn.Linear(16,16),nn.ReLU(),
          nn.Linear(16,8),nn.ReLU(),


          nn.Linear(8,8),nn.ReLU(),
          nn.Linear(8,4),nn.ReLU(),

          nn.Linear(4,4),nn.ReLU(),
          nn.Linear(4,2),nn.ReLU(),

          nn.Linear(2,1),nn.Tanh(),

          )

    def forward(self, x):

       	#print(x.shape)
       	x=self.convs(x)
        #print(x.shape)

        x=x.view(x.shape[0],256)
        #print(x.shape)
        x = self.lins(x)
        #print(x.shape)
        return x
    
    
# if __name__=="__main__":
# 	model=Net()
# 	print('Getting DATA')
# 	X,_=getXY()
# 	X=torch.Tensor(X)

# 	for dat in X:
# 		x=dat.reshape(1,5,8,8)
# 		y=model(x)
# 		break
