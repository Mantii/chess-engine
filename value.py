import chess
from model import Net
import torch
from data import toNumpy

def value(board):
	model = Net().cuda()
	checkpoint = torch.load('../2/modelsave.tar')
	model.load_state_dict(checkpoint['vae'])
	x=torch.Tensor(toNumpy(board)).view(1,5,8,8).cuda()
	y=model(x)
	return y

if __name__=="__main__":
	board=chess.Board()
	val=value(board)
	print(val)
	print(board.legal_moves)