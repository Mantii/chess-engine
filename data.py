import chess
import chess.pgn
import numpy as np
from tqdm import tqdm

#Load all available games from a PGN file
def load_data(n=0):
	pgn=open("data.pgn")
	games=[]
	i=0

	while 1:

		try:
			game=chess.pgn.read_game(pgn)

			if game==None:
				break

			games.append(game)
			i=i+1

			if i%1000==0:
				print('Samples Collected: ',i)

			if n!=0 and i==n:
				break

		except:
			break

	print('Samples Output: ', len(games))
	return games

# transforming all the data into input and a label
def getXY(n=0):
	games=load_data(n)
	Y=[]
	X=[]
	for game in tqdm(games,leave=False):
		y={'1/2-1/2':0,'1-0':1,'0-1':-1}[game.headers['Result']]
		board=game.board()
		for move in game.mainline_moves():
			x=toNumpy(board)
			X.append(x.reshape(1,5,8,8))
			Y.append(y)
			board.push(move)

	return X,Y
def avgwinrate(n=0):
	games=load_data(2000)
	s=0
	i=0
	for game in tqdm(games):
		y={'1/2-1/2':0.5,'1-0':1,'0-1':0}[game.headers['Result']]
		s+=y
		i+=1

	return s/i
			
#Conversion of the chess board into a numpy array
def toNumpy(board):
	b_str,turn=str(board.unicode).split("'")[1].split(' ')[0:2]
	turn={'b':-1,'w':1}[turn]

	H1=int(bool(board.castling_rights & chess.BB_H1))
	H8=int(bool(board.castling_rights & chess.BB_H8))
	A1=int(bool(board.castling_rights & chess.BB_A1)) 
	A8=int(bool(board.castling_rights & chess.BB_A8))


	x=np.zeros(64*5).reshape(64,5)
	x[:,4]=turn

	pieces={
		'P':[1,0,0,0,turn],
		'N':[0,1,0,0,turn],
		'B':[1,1,0,0,turn],
		'R':[1,0,1,0,turn],
		'Rc':[1,0,1,1,turn],
		'Q':[0,1,1,0,turn],
		'K':[1,1,1,0,turn],

		'p':[-1,0,0,0,turn],
		'n':[0,-1,0,0,turn],
		'b':[-1,-1,0,0,turn],
		'r':[-1,0,-1,0,turn],
		'rc':[-1,0,-1,-1,turn],
		'q':[0,-1,-1,0,turn],
		'k':[-1,-1,-1,0,turn]
	}
	pos=0
	for i in b_str:
		if i=='/':
			continue
		try:
			pos+=int(i)
		except:
			if pos==0 and A8==1:
				x[pos]=pieces['rc']
			elif pos==7 and H8==1:
				x[pos]=pieces['rc']
			elif pos==56 and A1==1:
				x[pos]=pieces['Rc']
			elif pos==63 and H1==1:
				x[pos]=pieces['Rc']
			else:
				x[pos]=pieces[i]
			pos+=1
	return x.reshape(5,8,8)









if __name__=="__main__":
	print(avgwinrate())