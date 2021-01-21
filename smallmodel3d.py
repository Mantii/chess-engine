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

       	self.convs=nn.Sequential(
       		nn.Conv3d(1,8,(5,4,4),padding=(2,0,0)),nn.BatchNorm3d(8),nn.ReLU(),
       		nn.Conv3d(8,8,(5,1,1),padding=(2,0,0)),nn.BatchNorm3d(8),nn.ReLU(),

          nn.Conv3d(8,16,(5,2,2),padding=(2,0,0)),nn.BatchNorm3d(16),nn.ReLU(),
       		nn.Conv3d(16,16,(5,1,1),padding=(2,0,0)),nn.BatchNorm3d(16),nn.ReLU(),
       		
          nn.Conv3d(16,32,(5,2,2),padding=(2,0,0)),nn.BatchNorm3d(32),nn.ReLU(),
       		nn.Conv3d(32,32,(5,1,1),padding=(2,0,0)),nn.BatchNorm3d(32),nn.ReLU(),
       		
          nn.Conv3d(32,64,(5,2,2),padding=(2,0,0)),nn.BatchNorm3d(64),nn.ReLU(),
       		nn.Conv3d(64,64,(5,2,2),padding=(2,0,0)),nn.BatchNorm3d(64),nn.ReLU(),
          nn.AvgPool3d((5,1,1))
       		)
        self.lins=nn.Sequential(


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
        # for i,lay in enumerate(self.convs):
        #   print(i,x.shape)
        #   x=lay(x)

       	#print(x.shape)
       	x=self.convs(x)
        #print(x.shape)

        x=x.view(x.shape[0],64)
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
