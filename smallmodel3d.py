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

       	x=self.convs(x)

        x=x.view(x.shape[0],64)
        x = self.lins(x)

        return x
    
