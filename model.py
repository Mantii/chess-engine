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

        #400
        self.c1=nn.Conv2d(5,400,kernel_size=(4,4))
        self.c2=nn.Conv2d(400,400,kernel_size=(1,1))
        self.c3=nn.Conv2d(400,400,kernel_size=(1,1))

        #800
        self.c4=nn.Conv2d(400,800,kernel_size=(4,4))
        self.c5=nn.Conv2d(800,800,kernel_size=(1,1))
        self.c6=nn.Conv2d(800,800,kernel_size=(1,1))

        #1600
        self.c7=nn.Conv2d(800,1600,kernel_size=(2,2))

        self.l1=nn.Linear(1600,1)

    def forward(self, x):

        x = F.relu(self.c1(x))
        #print(x.shape)
        x = F.relu(self.c2(x))
        #print(x.shape)
        x = F.relu(self.c3(x))
        #print(x.shape)
        x = F.relu(self.c4(x))
        #print(x.shape)
        x = F.relu(self.c5(x))
        #print(x.shape)
        x = F.relu(self.c6(x))
        #print(x.shape)
        x = F.relu(self.c7(x))
        #print(x.shape)

        x=x.view(x.shape[0],1600)
        #print(x.shape)
        x = self.l1(x)
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
