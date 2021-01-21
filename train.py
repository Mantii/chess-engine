
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import DataLoader
from data import getXY
from smallmodel3d import Net
from torch.utils.data import Dataset
from torch.autograd import Variable


class MyDataset(Dataset):
    def __init__(self, dataset1, dataset2):
        self.dataset1 = dataset1 # datasets should be sorted!
        self.dataset2 = dataset2

    def __getitem__(self, index):
        x1 = self.dataset1[index]
        x2 = self.dataset2[index]

        return x1, x2

    def __len__(self):
        return len(self.dataset1) # assuming both datasets have same length




num_epochs=10000
batch_size=128
saveN=5




num_batches=9852

# print("Load Data")
# X,Y=getXY(10000)

# print("\nRetrieved {} Board Positions".format(len(X)))
# dataset = MyDataset(torch.FloatTensor(X), torch.FloatTensor(Y))
# trainset = DataLoader(dataset, batch_size=batch_size,shuffle=True, num_workers=8,drop_last=True)
# torch.save(trainset,'data10k.pt')
# exit()
trainset = torch.load('data10k.pt')
model=Net().cuda()

criterion = nn.MSELoss(reduction='mean')

optimizer = optim.Adam(model.parameters(), lr=5e-4)

checkpoint = torch.load('../2/smallmodelsave.tar')
model.load_state_dict(checkpoint['vae'])
#optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
print("\nLoaded model checkpoint\nEpoch: ",checkpoint['epoch'],'\nLoss:',checkpoint['loss'])

for epoch in range(num_epochs):
	total_loss=0
	for data in tqdm(trainset,leave=False):
		x,y=data

		x=Variable(x).cuda()
		y=y.view(y.shape[0],1).cuda()
		#zero grads
		optimizer.zero_grad()

		#put data through layers
		output=model(x)

		#total loss
		loss=criterion(output,y)
		loss.backward()
		optimizer.step()
		total_loss+=float(loss.data)
	total_loss/=num_batches
	print('[{}/{}] \t L:{:.7f}'.format(epoch , num_epochs, total_loss))
	if epoch%saveN==0 and epoch!=0:
		torch.save({
	        'vae': model.state_dict(),
	        'optimizer_state_dict': optimizer.state_dict(),
	        'epoch':epoch,#+checkpoint['epoch']+1,
	        'loss':total_loss
	        },'smallmodelsave.tar')
		print('\nModel Saved')
