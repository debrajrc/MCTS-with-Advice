"""! @brief Classes to create MDP for pacman."""

import sys, json, os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from adviceMCTS.Examples.pacman.pacmanStormValueGen import getValue
from adviceMCTS.Examples.pacman.pacmanPrismNoFood import pacmanEngine as pacmanEngineNoFood

WORK_DIR = os.environ.get('GLOBALSCRATCH')
if WORK_DIR == None:
	WORK_DIR = ""
else:
	WORK_DIR = WORK_DIR + os.sep

# CURRENTPWD = os.path.dirname(os.path.abspath(__file__))
# SRC_DIR = os.path.join(CURRENTPWD, '../src')
# sys.path.append(SRC_DIR)

from adviceMCTS.simulationClasses import *
import adviceMCTS.util as util

TEMP_DIR = WORK_DIR+'tempFiles'+os.sep+'prism'
util.mkdir(TEMP_DIR)

def readFromFile(fileName):
	"""! Given a prism file, runs MCTS with it

	@param fileName location to a text file describing the layout

	@return X,Y dimension of the grid
	@return numGhosts number of ghosts
	@return layoutText "%" for walls, "." for food
	@return agentInfo (((x,y),d) for each agents)
	d = 0 --> no direction
	d = 1 --> East
	d = 2 --> West
	d = 3 --> North
	d = 4 --> South
	"""
	f = open(fileName)
	sizeStr = f.readline().split()
	if len(sizeStr) > 3:
		raise Exception("Size mismatch : "+str(sizeStr))
	else:
		X = int(sizeStr[0])
		Y = int(sizeStr[1])
		numGhosts = int(sizeStr[2])
	layoutText = []
	for i in range(X):
		layoutText.append(f.readline().strip('\n'))
	agentInfo = []
	for i in range(numGhosts+1):
		agentStr = f.readline().split()
		if len(agentStr) > 3:
			raise Exception("Size mismatch : "+str(agentStr))
		else:
			agentInfo.append(((int(agentStr[0]),int(agentStr[1])),int(agentStr[2])))
	f.close()
	return (X,Y,numGhosts,layoutText,agentInfo)

class Layout:
	##
	# A Layout manages the static information about the game board.
	#
	# X Y numGhosts\n
	# Foods and walls in a layout file with height X and width Y :\n
	# (0,0)     (0,1)   .  .  .  (0,Y-1)\n
	# (1,0)     (1,1)               .\n
	#  .                .           .\n
	#  .                  .         .\n
	#  .                    .       .\n
	# (X-1,0)   (X-1,1)  . . . (X-1,Y-1)\n
	#  x y direction for each agents (agent 0 is pacman)


	def __init__(self,X,Y,numGhosts,layoutText,agentInfo):
		self.width = Y
		self.height= X
		self.walls = [[False for j in range(self.width)] for i in range(self.height)]
		self.food = [[False for j in range(self.width)] for i in range(self.height)]
		self.capsules = []
		self.pacmanPositions = []
		self.ghostPositions = []
		self.ghostDirection = []
		self.numGhosts = numGhosts
		self.totalFood = 0
		self.processAgents(agentInfo)
		self.processLayoutText(layoutText)
		self.layoutText = layoutText

	def __str__(self):
		return "\n".join(self.layoutText)+"\n"+str(self.pacmanPositions)+"\n"+str(self.ghostPositions)+"\n"+str(self.ghostDirection)

	def deepCopy(self):
		return Layout(self.width, self.height, self.numGhosts, self.layoutText[:], self.agentInfo[:])

	def processAgents(self, agentInfo):
		self.pacmanPositions = []
		self.ghostPositions = []
		self.ghostDirection = []
		self.pacmanPositions.append(agentInfo[0][0])
		for g in agentInfo[1:]:
			self.ghostPositions.append(g[0])
			self.ghostDirection.append(g[1])
		self.numGhosts = len(self.ghostPositions)
	def processLayoutText(self, layoutText):
		# maxY = self.height - 1
		for x in range(self.height):
			for y in range(self.width):
				layoutChar = layoutText[x][y]
				self.processLayoutChar(x, y, layoutChar)
		# self.agentPositions.sort()
		# self.agentPositions = [ ( i == 0, pos) for i, pos in self.agentPositions]

	def processLayoutChar(self, x, y, layoutChar):
		if layoutChar == '%':
			self.walls[x][y] = True
		elif layoutChar == '.':
			self.food[x][y] = True
			self.totalFood += 1
		# elif layoutChar == 'o':
		# 	self.capsules.append((x, y))
		# elif layoutChar == 'P':
		# 	self.pacmanPositions.append((x, y))
		# elif layoutChar == 'G':
		# 	self.ghostPositions.append((x, y))
		# 	self.numGhosts += 1
		# 	self.ghostDirection.append(0)
		# elif layoutChar in ['1','2','3','4']:
		# 	self.ghostPositions.append((x, y))
		# 	self.numGhosts += 1
		# 	self.ghostDirection.append(int(layoutChar))

	def getArray(self): # returns tensor with 6 channels
		X = len(self.walls)
		Y=len(self.walls[0])
		array = np.zeros((6,X,Y)) # layers: walls, pacman, east ghosts, west ghosts, north ghosts, south ghosts
		for i in range(X):
			for j in range(Y):
				if self.walls[i][j]:
					array[0][i][j] = 1
		array[1][self.pacmanPositions[0][0]][self.pacmanPositions[0][1]] = 1
		for g in range(self.numGhosts):
			# if self.ghostDirection[g] == 0:
			# 	for i in range(2,6):
			# 		array[i][self.ghostPositions[g][0]][self.ghostPositions[g][1]] = 1
			# else:
			array[self.ghostDirection[g]+1][self.ghostPositions[g][0]][self.ghostPositions[g][1]] = 1
		return(array)

	def getArrayWithFood(self): # returns tensor with 7 channels
		X = len(self.walls)
		Y=len(self.walls[0])
		array = np.zeros((7,X,Y)) # layers: walls, pacman, east ghosts, west ghosts, north ghosts, south ghosts, foods
		for i in range(X):
			for j in range(Y):
				if self.walls[i][j]:
					array[0][i][j] = 1
				if self.food[i][j]:
					array[6][i][j] = 1
		array[1][self.pacmanPositions[0][0]][self.pacmanPositions[0][1]] = 1
		for g in range(self.numGhosts):
			# if self.ghostDirection[g] == 0:
			# 	for i in range(2,6):
			# 		array[i][self.ghostPositions[g][0]][self.ghostPositions[g][1]] = 1
			# else:
			array[self.ghostDirection[g]+1][self.ghostPositions[g][0]][self.ghostPositions[g][1]] = 1
		return(array)

##
# class to create a prism file for pacman and do operations
class pacmanEngine:

	def __init__(self,X,Y,numGhosts,layoutText,agentInfo, fname=None):  #drawDepth, timermax, scaredFactor,

		self.layout = Layout(X,Y,numGhosts,layoutText,agentInfo) # in layout row and columns are counted from bottom left corner
		self.fname = fname
		self.width = self.layout.width
		self.height = self.layout.height
		self.walls = self.layout.walls
		self.food = self.layout.food
		# self.drawDepth = drawDepth
		self.pacmanPosition = self.layout.pacmanPositions[0] # Assuming 1 pacman
		self.ghostPositions = self.layout.ghostPositions # of form [(xg,yg),...]
		self.ghostDirection = self.layout.ghostDirection
		self.numGhosts = self.layout.numGhosts
		# self.timermax = timermax # timer for ghosts
		# self.scaredFactor = scaredFactor # factor by which ghosts gets slower if scared
		# self.capsules = self.layout.capsules

	def __str__(self):
		return (str(self.layout))

	def _openOutput(self):
		if self.fname == None:
			self.out = sys.stdout
		else:
			file = open(self.fname,'w+')
			self.out = file

	# def getTraceDepth(self, horizon):  # since each step is (num of modules) length long
		# return (horizon*(self.numGhosts*3+2)+1)

	def _ghostDirections(self,g):
		stringEast = 'formula ghostEast%s = '%g
		stringWest = 'formula ghostWest%s = '%g
		stringNorth = 'formula ghostNorth%s = '%g
		stringSouth = 'formula ghostSouth%s = '%g
		for i in range(1,self.height-1):
			for j in range(1,self.width-1):
				stringEast+=f'(x{i}_{j}=xg{g} & y{i}_{j}=yg{g} & w{i}_{j+1}=0) | '
				stringWest+=f'(x{i}_{j}=xg{g} & y{i}_{j}=yg{g} & w{i}_{j-1}=0) | '
				stringNorth+=f'(x{i}_{j}=xg{g} & y{i}_{j}=yg{g} & w{i-1}_{j}=0) | '
				stringSouth+=f'(x{i}_{j}=xg{g} & y{i}_{j}=yg{g} & w{i+1}_{j}=0) | '
		stringEast = stringEast[:-3]+';\n'
		stringWest = stringWest[:-3]+';\n'
		stringNorth = stringNorth[:-3]+';\n'
		stringSouth = stringSouth[:-3]+';\n'
		print(stringEast,file=self.out)
		print(stringWest,file=self.out)
		print(stringNorth,file=self.out)
		print(stringSouth,file=self.out)
		# print(f'formula ghostOpenEast{g} = ghostEast{g} ? 1 : 0;',file=self.out)
		# print(f'formula ghostOpenWest{g} = ghostWest{g} ? 1 : 0;',file=self.out)
		# print(f'formula ghostOpenNorth{g} = ghostNorth{g} ? 1 : 0;',file=self.out)
		# print(f'formula ghostOpenSouth{g} = ghostSouth{g} ? 1 : 0;',file=self.out)
		# print(f'formula ghostOpen{g} = ghostOpenEast{g} + ghostOpenWest{g} + ghostOpenNorth{g} + ghostOpenSouth{g};\n',file=self.out) # number of open directions (> 2 -> crossing)
		# can only change direction if the direction is open and the ghost is not going opposite direction
		print(f'formula ghostLegalEast{g} = (d{g} != 2) & ghostEast{g};', file=self.out)
		print(f'formula ghostLegalWest{g} = (d{g} != 1) & ghostWest{g};', file=self.out)
		print(f'formula ghostLegalNorth{g} = (d{g} != 4) & ghostNorth{g};', file=self.out)
		print(f'formula ghostLegalSouth{g} = (d{g} != 3) & ghostSouth{g};', file=self.out)
		print(f'formula ghostLegalIntEast{g} = ghostLegalEast{g} ? 1 : 0;',file=self.out)
		print(f'formula ghostLegalIntWest{g} = ghostLegalWest{g} ? 1 : 0;',file=self.out)
		print(f'formula ghostLegalIntNorth{g} = ghostLegalNorth{g} ? 1 : 0;',file=self.out)
		print(f'formula ghostLegalIntSouth{g} = ghostLegalSouth{g} ? 1 : 0;',file=self.out)
		print(f'formula ghostLegal{g} = ghostLegalIntEast{g} + ghostLegalIntWest{g} + ghostLegalIntNorth{g} + ghostLegalIntSouth{g};',file=self.out) # number of legal directions
		print(f'formula ghostLegalRatioEast{g} = ghostLegalIntEast{g}/ghostLegal{g};',file=self.out)
		print(f'formula ghostLegalRatioWest{g} = ghostLegalIntWest{g}/ghostLegal{g};',file=self.out)
		print(f'formula ghostLegalRatioNorth{g} = ghostLegalIntNorth{g}/ghostLegal{g};',file=self.out)
		print(f'formula ghostLegalRatioSouth{g} = ghostLegalIntSouth{g}/ghostLegal{g};',file=self.out)
		print(f'formula isEaten{g} = (xg{g} = xp & yg{g} = yp);',file=self.out)
		print(f'formula isExchange{g} = (lastxg{g} = xp & lastyg{g} = yp & xg{g} = lastxp & yg{g} = lastyp);',file=self.out)
		print(f'formula isLoss{g} = (isEaten{g} | isExchange{g});',file=self.out)

	def _totalFood(self):
		totalFood = 'formula totalFood ='
		for i in range(self.height): # i is counted from top to bottom of grid starting from 0
			for j in range(self.width): # j is counted from left to right of grid starting from 0
				totalFood+=f' f{i}_{j} +'
		totalFood = totalFood[:-1]+';'
		print(totalFood, file=self.out)

	def _initialize(self):
		print('mdp\n',file=self.out)
		for i in range(self.height): # # i is counted from top to bottom of grid starting from 0
			for j in range(self.width): # j is counted from left to right of grid starting from 0
				# constants for co-ordinates
				print(f'const int x{i}_{j} = {i};',file=self.out) # ints for every x-position in the grid
				print(f'const int y{i}_{j} = {j};',file=self.out) # ints for every y-position in the grid
				# walls
				print(f'const int w{i}_{j} = {[0,1][self.walls[i][j]]};',file=self.out) # ints for walls
		self._totalFood()
		# print('const int drawDepth = %s;' %self.drawDepth,file=self.out)
		isLossString = f'formula isLoss = (token = 0) & ('
		for g in range(self.numGhosts):
			self._ghostDirections(g)
			isLossString+=f'isLoss{g} | '
		isLossString = isLossString[:-3]+');'
		print(isLossString,file=self.out)
		print(f'formula isWin = (token = 0) & (totalFood = 0);',file=self.out)
		# print('formula isDraw = token = 0 &  steps = %s & !isWin & !isLoss;' %self.drawDepth,file=self.out)

	def _moduleArbiter(self):
		print('\nmodule arbiter\n',file=self.out)
		print(f'token : [0 .. {3+self.numGhosts}] init 0;',file=self.out)
		'''
		1 token to update the Win/Loss state
		1 token for pacman
		1 token for each ghosts
		1 token to update food and check at the end
		1 extra token to loop in the win/loss state (game is over)
		'''

		print('result : [0 .. 2] init 0;',file=self.out)
		'''
		0 = Game is not over yet
		1 = Win
		2 = Loss
		'''

		# print('steps : [0 .. %s] init 0;' %(self.drawDepth),file=self.out)

		# token = 1

		# actions for pacman
		print('',file=self.out)
		print('[East] (token = 1) -> 1: (token\' = 2);',file=self.out)
		print('[West] (token = 1) -> 1: (token\' = 2);',file=self.out)
		print('[North] (token = 1) -> 1: (token\' = 2);',file=self.out)
		print('[South] (token = 1) -> 1: (token\' = 2);',file=self.out)

		# token = 2 ... numGhosts+1
		# actions for ghosts
		print('',file=self.out)
		for i in range (self.numGhosts):
			print(f'[g{i}] (token = {i+2}) -> 1: (token\' = {i+3});' ,file=self.out)

		# token = numGhosts + 2
		# update food
		print(f'\n[updateFood] (token = {self.numGhosts+2}) -> 1: (token\' = 0);\n',file=self.out)#{self.numGhosts+3}
		# print(f'\n[updateWinLoss] (token = {self.numGhosts+2}) -> 1: (token\' = 0);\n',file=self.out)

		# token = 0
		# Check for win or loss
		print(f'[updateWinLoss] (isWin & token = 0) -> 1: (token\' = {self.numGhosts+3}) & (result\' = 1);',file=self.out)
		print(f'[updateWinLoss] (isLoss & token = 0) -> 1: (token\' = {self.numGhosts+3}) & (result\' = 2);',file=self.out)
		print(f'[updateWinLoss] (!(isLoss | isWin) & token = 0) -> 1: (token\' = 1);',file=self.out)

		# token = numGhosts + 3
		# self-loop after win or loss
		print(f'[] (token = {self.numGhosts+3}) -> 1: (token\' = token);',file=self.out)

		print('endmodule\n',file=self.out)

	def _pacmanMove(self,i,j):
		'''
		East  y'=y+1
		West  y'=y-1
		North   x'=x-1
		South   x'=x+1
		'''
		print(f'[East] (x{i}_{j}=xp & y{i}_{j}=yp & w{i}_{j+1}=0) -> 1: (xp\'=xp) & (yp\'=yp+1) & (lastxp\'=xp) & (lastyp\'=yp);', file=self.out)
		print(f'[West] (x{i}_{j}=xp & y{i}_{j}=yp & w{i}_{j-1}=0) -> 1: (xp\'=xp) & (yp\'=yp-1) & (lastxp\'=xp) & (lastyp\'=yp);', file=self.out)
		print(f'[North] (x{i}_{j}=xp & y{i}_{j}=yp & w{i-1}_{j}=0) -> 1: (yp\'=yp) & (xp\'=xp-1) & (lastxp\'=xp) & (lastyp\'=yp);', file=self.out)
		print(f'[South] (x{i}_{j}=xp & y{i}_{j}=yp & w{i+1}_{j}=0) -> 1: (yp\'=yp) & (xp\'=xp+1) & (lastxp\'=xp) & (lastyp\'=yp);', file=self.out)

	def _modulePacman(self):
		print('\nmodule pacman\n',file=self.out)
		print(f'xp: [1 .. {self.height-2}] init {self.pacmanPosition[0]};',file=self.out)
		print(f'yp: [1 .. {self.width-2}] init {self.pacmanPosition[1]};',file=self.out)
		print(f'lastxp: [0 .. {self.height-2}] init 0;',file=self.out)
		print(f'lastyp: [0 .. {self.width-2}] init 0;\n',file=self.out)

		for i in range(1,self.height-1):
			for j in range(1,self.width-1): # assuming there are walls in the border
				self._pacmanMove(i,j)
		print('endmodule\n',file=self.out)

	def _moduleGhost(self,g):
		print(f'\nmodule ghost{g}\n',file=self.out)

		print(f'xg{g}: [1 .. {self.height-2}] init {self.ghostPositions[g][0]};',file=self.out)
		print(f'yg{g}: [1 .. {self.width-2}] init {self.ghostPositions[g][1]};',file=self.out)
		print(f'lastxg{g}: [0 .. {self.height-2}] init 0;',file=self.out)
		print(f'lastyg{g}: [0 .. {self.width-2}] init 0;',file=self.out)
		'''
		Ghost last direction
		0  undefined
		1  east
		2  west
		3  north
		4  south
		'''
		print(f'd{g}: [0 .. 4] init {self.ghostDirection[g]};',file=self.out)

		#normal ghost
		print(f'[g{g}] true -> ghostLegalRatioEast{g} : (lastxg{g}\'=xg{g}) & (lastyg{g}\'=yg{g}) & (yg{g}\'=yg{g}+1) & (d{g}\'=1)+ ghostLegalRatioWest{g} : (lastxg{g}\'=xg{g}) & (lastyg{g}\'=yg{g}) & (yg{g}\'=yg{g}-1)  & (d{g}\'=2) + ghostLegalRatioNorth{g} : (lastxg{g}\'=xg{g}) & (lastyg{g}\'=yg{g}) & (xg{g}\'=xg{g}-1)  & (d{g}\'=3)+ ghostLegalRatioSouth{g} : (lastxg{g}\'=xg{g}) & (lastyg{g}\'=yg{g}) & (xg{g}\'=xg{g}+1) & (d{g}\'=4);',file=self.out)
		print('endmodule\n',file=self.out)

	def _initialFood(self):
		for i in range(self.height): # i is counted from left to right of grid starting from 0
			for j in range(self.width): # j is counted from bottom to top of grid starting from 0
				print(f'f{i}_{j} : [0 .. 1] init {[0,1][self.food[i][j]]};',file=self.out) # ints for food

	def _moduleUpdateFood(self):
		print('\nmodule check\n',file=self.out)
		print('foodEaten: [0 .. 1] init 0;',file=self.out) # update if a food or capsule is eaten in a step

		self._initialFood()

		for i in range(1,self.height-1):
			for j in range(1,self.width-1): # assuming there are walls in the border
				# eat food
				print(f'[updateFood] (x{i}_{j}=xp & y{i}_{j}=yp) -> 1: (f{i}_{j}\'=0) & (foodEaten\'=f{i}_{j});',file=self.out)
#                print('[update] true -> 1: (foodEaten\'=foodEaten);' ,file=self.out)
		print('endmodule\n',file=self.out)

	def _labels(self):
		print('\nlabel "Win" = (result = 1);',file=self.out)
		print('label "Loss" = (result = 2);',file=self.out)
		# print('label "Draw" = (result = 1);',file=self.out)

	def _rewards(self):
		print('\nrewards',file=self.out)
		# -1 for each move
		print('[East] true : -1;',file=self.out)
		print('[West] true: -1;',file=self.out)
		print('[North] true: -1;',file=self.out)
		print('[South] true: -1;',file=self.out)

		# reward for eating food
		print('(foodEaten = 1 & token = 0) : 10;' ,file=self.out)
		# print('(foodEaten = 2 & token = 0): 10;' ,file=self.out)
		# for i in range (self.numGhosts):
			# print('(eaten%s = 1 & token = %s): 10;' %(i,3*i+2),file=self.out)
		# reward for eating all food
		print('isWin: 500;',file=self.out)

		# penalty for dying
		print('isLoss: -500;',file=self.out)

		print('endrewards',file=self.out)

	def createPrismFile(self):
		self._openOutput()
		self._initialize()
		self._moduleArbiter()
		self._modulePacman()
		for g in range(self.numGhosts):
			self._moduleGhost(g)
		self._moduleUpdateFood()
		self._labels()
		self._rewards()
		if self.out != sys.stdout:
			self.out.close()
		return(self.fname)

	def savePickle(self):
		import pickle
		try:
			out = p.__dict__.pop('out')
		except:
			pass
		pickle.dump(p, open(str(self.fname)+'_save.p', 'wb'))
		try:
			p.out = out
		except:
			pass

	def printLayout(self,d):
			# s=s.split('__')
			# d = json.loads(s[1])
			# if d['token'] not in [0,(2+self.numGhosts)]:
				# return ''
			s = ''
			currentLayout = [[False for j in range(self.width)] for i in range(self.height)]
			for i in range(self.height):
				for j in range(self.width):
					if self.walls[i][j]:
						currentLayout[i][j] = '%'
					else:
						foodKey = 'f'+str(i)+'_'+str(j)
						if d[foodKey] == 1:
							currentLayout[i][j] = '.'
						# elif d[foodKey] == 2:
						#     currentLayout[i][j] = 'o'
						else:
							currentLayout[i][j] = ' '
			pacmanX, pacmanY = d['xp'], d['yp']
			currentLayout[pacmanX][pacmanY] = 'P'
			for g in range(self.numGhosts):
				ghostX, ghostY = d['xg'+str(g)], d['yg'+str(g)]
				currentLayout[ghostX][ghostY] = 'G'
			for line in currentLayout:
				for l in line:
					s+=str(l)
				s+='\n'
			return s

	def strLayoutFile(self,d):
		s = f"{self.height} {self.width} {self.numGhosts}\n"
		# currentLayout = [[False for j in range(self.width)] for i in range(self.height)]
		for i in range(self.height):
			line = ""
			for j in range(self.width):
				if self.walls[i][j]:
					line += '%'
				else:
					foodKey = 'f'+str(i)+'_'+str(j)
					if d[foodKey] == 1:
						line += '.'
					else:
						line += ' '
			s+=line+'\n'
		pacmanX, pacmanY = d['xp'], d['yp']
		s += f"{pacmanX} {pacmanY} 0\n"
		for g in range(self.numGhosts):
			ghostX, ghostY, ghostDir = d['xg'+str(g)], d['yg'+str(g)], d['d'+str(g)]
			s+=f"{ghostX} {ghostY} {ghostDir}\n"
		return s

	def getInfo(self,d): #returns X,Y,numGhosts,layoutText,agentInfo from state description
		X = self.height
		Y = self.width
		numGhosts = self.numGhosts
		layoutText = []
		for i in range(self.height):
			line = ""
			for j in range(self.width):
				if self.walls[i][j]:
					line += '%'
				else:
					foodKey = 'f'+str(i)+'_'+str(j)
					if d[foodKey] == 1:
						line += '.'
					else:
						line += ' '
			layoutText.append(line)
		agentInfo = []
		pacmanX, pacmanY = d['xp'], d['yp']
		agentInfo.append(((pacmanX,pacmanY),0))
		for g in range(self.numGhosts):
			ghostX, ghostY, ghostDir = d['xg'+str(g)], d['yg'+str(g)], d['d'+str(g)]
			agentInfo.append(((ghostX,ghostY),ghostDir))
		return X,Y,numGhosts,layoutText,agentInfo

	def getScore(self,d,coef=0.5): # we had coefs alpha and beta in the old implem. the coef should be beta/(1-alpha).
		ghostValuation = 0
		(xp,yp)=d['xp'], d['yp']#position of Pacman
		X=self.height
		Y=self.width
		# towardsGhosts = 1
		for ghost in range(self.numGhosts):
			(xg,yg) = d['xg'+str(ghost)], d['yg'+str(ghost)]
		#     ghostDirection = d['d'+str(ghost)]
		#     ghostClose = True
		#     if ghostDirection == 1 and yg > yp:
		#         ghostClose = False
		#     if ghostDirection == 2 and yg < yp:
		#         ghostClose = False
		#     if ghostDirection == 3 and xg < xp:
		#         ghostClose = False
		#     if ghostDirection == 4 and xg > xp:
		#         ghostClose = False
		#     if ghostClose:
			distance=abs(xg-xp)+abs(yg-yp)# distance between ghost number ghost and pacman
			ghostValuation += -1/(1+distance)
		#         towardsGhosts += 1
		if self.numGhosts>0:
			ghostValuation /= self.numGhosts
		foodValuation = 0
		numFoods = 0
		for i in range(X):
			for j in range(Y):
				if 'f'+str(i)+'_'+str(j) in d:
					if d['f'+str(i)+'_'+str(j)]==1: # is there food in (i, j)?
						numFoods+=1
						distance=abs(i-xp)+abs(j-yp)# distance between pacman and (i,j)
						foodValuation += -1/(1+distance)
		if numFoods!=0:
			foodValuation /= numFoods
		return coef*ghostValuation-(1-coef)*(foodValuation)

	def getArray(self,d):
		X = len(self.walls)
		Y=len(self.walls[0])
		array = np.zeros((6,X,Y)) # layers: walls, pacman, east ghosts, west ghosts, north ghosts, south ghosts
		for i in range(X):
			for j in range(Y):
				if self.walls[i][j]:
					array[0][i][j] = 1
		(xp,yp)=d['xp'], d['yp']
		array[1][xp][yp] = 1
		for g in range(self.numGhosts):
			(xg,yg) = d['xg'+str(g)], d['yg'+str(g)]
			ghostDirection = d['d'+str(g)]
			if ghostDirection == 0:
				for i in range(2,6):
					array[i][xg][yg] = 1
			else:
				array[ghostDirection+1][xg][yg] = 1
		return(array)

	def getArrayWithFoodModel(self,d,model):
		X = len(self.walls)
		Y=len(self.walls[0])
		array = np.zeros((8,X,Y)) # layers: walls, pacman, east ghosts, west ghosts, north ghosts, south ghosts, foods, result from other NN
		for i in range(X):
			for j in range(Y):
				if self.walls[i][j]:
					array[0][i][j] = 1
				else:
					foodKey = 'f'+str(i)+'_'+str(j)
					if d[foodKey] == 1:
						array[6][i][j] = 1
		(xp,yp)=d['xp'], d['yp']
		array[1][xp][yp] = 1
		for g in range(self.numGhosts):
			(xg,yg) = d['xg'+str(g)], d['yg'+str(g)]
			ghostDirection = d['d'+str(g)]
			if ghostDirection == 0:
				for i in range(2,6):
					array[i][xg][yg] = 1
			else:
				array[ghostDirection+1][xg][yg] = 1
		arrayWithoutFood = np.expand_dims(array[:6],axis=0)
		actionValues = keras.backend.get_value(model(arrayWithoutFood))[0]
		maxValue = max(actionValues)
		array[7][xp][yp] = maxValue
		return(array)

	def getArrayWithFood(self,d):
		X = len(self.walls)
		Y=len(self.walls[0])
		array = np.zeros((7,X,Y)) # layers: walls, pacman, east ghosts, west ghosts, north ghosts, south ghosts, foods
		for i in range(X):
			for j in range(Y):
				if self.walls[i][j]:
					array[0][i][j] = 1
				else:
					foodKey = 'f'+str(i)+'_'+str(j)
					if d[foodKey] == 1:
						array[6][i][j] = 1
		(xp,yp)=d['xp'], d['yp']
		array[1][xp][yp] = 1
		for g in range(self.numGhosts):
			(xg,yg) = d['xg'+str(g)], d['yg'+str(g)]
			ghostDirection = d['d'+str(g)]
			if ghostDirection == 0:
				for i in range(2,6):
					array[i][xg][yg] = 1
			else:
				array[ghostDirection+1][xg][yg] = 1
		return(array)

	def getLayoutDescription(self):
		X = len(self.walls)
		Y=len(self.walls[0])
		return (X,Y,self.numGhosts,self.walls)

##
# prints layout from a numpy array
def getLayoutFromArray(a):
	with np.printoptions(threshold=np.inf):
		print(a)
	size = a.shape
	X = size[2]
	Y = size[3]
	layout = [[' ' for j in range(Y)] for i in range(X)]
	for i in range(X):
		for j in range(Y):
			if a[0][0][i][j] == 1:
				layout[i][j] = '%'
	for i in range(X):
		for j in range(Y):
			if a[0][-1][i][j] == 1:
				layout[i][j] = '.'
	s = "\n".join(["".join(layout[i]) for i in range(X)])
	for k in range(1,size[1]-1):
		for i in range(X):
			for j in range(Y):
				if a[0][k][i][j] == 1:
					s+="\n"+str(i)+" "+str(j)+" "+str(k-1)
	s+="\n"
	return s


def createEngine(fname):
	"""! create a pacmanEngine from a text file
	"""
	X,Y,numGhosts,layoutText,agentInfo = readFromFile(fname)
	prismFile = TEMP_DIR+os.sep+fname.split(os.sep)[-1][:-4]+'_'+str(os.getpid())+'.nm'
	p = pacmanEngine(X,Y,numGhosts,layoutText,agentInfo,prismFile)
	return(p)


# class MDPStateScoreDistance(MDPStateScoreInterface):
# 	def __init__(self,scoreFunction):
# 		self.scoreFunction = scoreFunction
#
# 	def getScore(self, executionEngine):
# 		endState = executionEngine.mdpEndState()
# 		stateDescription = executionEngine.mdpOperations.stateDescription(endState)
# 		return (self.scoreFunction(stateDescription))


class MDPStateScoreDistanceOld(MDPStateScoreInterface):
	"""! Distance function that uses a linear combination of distance (exact distance in the graph) from ghosts and remaining foods
	"""
	def __init__(self,X,Y,numGhosts,walls):
		self.X=X
		self.Y=Y
		self.numghosts=numGhosts
		self.walls=walls

	def gridDistance(self,posList):
		infty=self.X*self.Y+1
		r = [[ infty for j in range(self.Y)] for i in range(self.X)]
		queue = []
		maxd=infty
		for ((x,y),direction) in posList:
			r[x][y]=0
			# queue.append((x,y,0))
			maxd=0
			(xr,yr)=(x,y)
			if direction==0:
				(xr,yr)=(-1,-1)
			if direction==1:
				yr-=1
			elif direction==2:
				yr+=1
			if direction==3:
				xr+=1
			elif direction==4:
				xr-=1
			for xx,yy in [(x-1,y),(x,y-1),(x+1,y),(x,y+1)]:
				if xx>=0 and xx<self.X and yy>=0 and yy<self.Y and (not self.walls[xx][yy]) and r[xx][yy]>1 and (xx,yy)!=(xr,yr):
					r[xx][yy]=1
					queue.append((xx,yy,1))
					maxd=1
		while len(queue)>0:
			x,y,d = queue.pop(0)
			for xx,yy in [(x-1,y),(x,y-1),(x+1,y),(x,y+1)]:
				if xx>=0 and xx<self.X and yy>=0 and yy<self.Y and (not self.walls[xx][yy]) and r[xx][yy]>d+1:
					r[xx][yy]=d+1
					queue.append((xx,yy,d+1))
					if d+1>maxd:
						maxd=d+1
		# print(f"Distance matrix from positions {posList}")
		# for xx in range(self.X):
		#     for yy in range(self.Y):
		#         if r[xx][yy] == infty:
		#             print(' %  ',end="")
		#         else:
		#             print(str(r[xx][yy]).zfill(3)+" ",end="")
		#     print("")
		# print("maxd",maxd)
		return (r,maxd)
	def getScore(self, executionEngine):
		endState = executionEngine.mdpEndState()
		stateDescription = executionEngine.mdpOperations.stateDescription(endState)

		coef = 0.5 # we had coefs alpha and beta in the old implem. the coef should be beta/(1-alpha).
		ghostValuation = 0
		numfoods=0#number of food left to eat
		pacmanPos=stateDescription['xp'], stateDescription['yp']#position of Pacman
		ghostList=[]
		for ghost in range(self.numghosts):
			ghostPos = stateDescription['xg'+str(ghost)], stateDescription['yg'+str(ghost)]
			ghostDirection = stateDescription['d'+str(ghost)]
			ghostList.append((ghostPos,ghostDirection))
		pacmanDistance,maxPacmanDist=self.gridDistance([(pacmanPos,0)])
		ghostsDistanceList=[self.gridDistance([ghostList[ghost]]) for ghost in range(self.numghosts)]
		for ghost in range(self.numghosts):
			d=ghostsDistanceList[ghost][0][pacmanPos[0]][pacmanPos[1]]# distance between ghost number ghost and pacman
			ghostValuation += -100/(1+d)
		if self.numghosts>0:
			ghostValuation /= self.numghosts
		foodValuation = 0
		for i in range(self.X):
			for j in range(self.Y):
				if 'f'+str(i)+'_'+str(j) in stateDescription:
					if stateDescription['f'+str(i)+'_'+str(j)]==1: # is there food in (i, j)?
						numfoods+=1
						d=pacmanDistance[i][j]# distance between pacman and (i,j)
						foodValuation += -100/(1+d)
		if numfoods!=0:
			foodValuation /= numfoods
		return coef*ghostValuation-(1-coef)*(foodValuation)

class MDPStateScoreMctsNN(MDPStateScoreInterface):
	"""! State score using neural network
	@params pacmanEngine needed to create array from an executionEngine
	@params model neural network model
	@params minY
	@params maxY
	@return (value given by NN)*(maxY-minY)+minY
	"""
	def __init__(self,pacmanEngine,model,minY,maxY):
		self.pacmanEngine = pacmanEngine
		self.model=model
		self.minY=minY
		self.maxY=maxY

	def getScore(self, executionEngine):
		endState = executionEngine.mdpEndState()
		stateDescription = executionEngine.mdpOperations.stateDescription(endState)
		# result = [p.name for p in executionEngine.result()]
		# if "Win" in result:
		# 	return 500
		# elif "Loss" in result:
		# 	return -500
		X,Y,numGhosts,walls = self.pacmanEngine.getLayoutDescription()
		pacmanPos=stateDescription['xp'], stateDescription['yp']#position of Pacman
		lastPacmanPos=stateDescription['lastxp'], stateDescription['lastyp']
		for ghost in range(numGhosts):
			ghostPos = stateDescription['xg'+str(ghost)], stateDescription['yg'+str(ghost)]
			lastGhostPos = stateDescription['lastxg'+str(ghost)], stateDescription['lastyg'+str(ghost)]
			# print(pacmanPos,ghostPos,lastPacmanPos,lastGhostPos)
			if pacmanPos == ghostPos or (lastPacmanPos == ghostPos and pacmanPos == lastGhostPos):
				return -500
		x = np.expand_dims(self.pacmanEngine.getArrayWithFood(stateDescription),axis=0)
		x = tf.transpose(x, [0, 2, 3, 1])
		value = keras.backend.get_value(self.model(x))[0][0]
		return value*(self.maxY-self.minY)+self.minY

class MDPStateScoreSafetyNN(MDPStateScoreInterface):
	def __init__(self,pacmanEngine,model):
		self.pacmanEngine = pacmanEngine
		self.model=model

	def getScore(self, executionEngine):
		endState = executionEngine.mdpEndState()
		stateDescription = executionEngine.mdpOperations.stateDescription(endState)
		x = np.expand_dims(self.pacmanEngine.getArray(stateDescription),axis=0)
		x = tf.transpose(x, [0, 2, 3, 1]) # assuming neural network takes input of the form NHWC
		value = keras.backend.get_value(self.model(x))[0][0]
		return 500*(value-1)

class MDPNNActionAdvice( MDPActionAdviceInterface[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction] ):
	"""! action advice that returns action over a given threshold
	@params model NN model
	@params transformer a sklearn.preprocessing.QuantileTransformer object
	@params pacmanEngine
	@params threshold

	@returns choices set of actions
	"""

	def __init__(self,model,transformer,pacmanEngine,threshold):
		self.model = model
		self.transformer = transformer
		self.pacmanEngine = pacmanEngine
		self.threshold = threshold

	def deepCopy(self):
		return MDPNNActionAdvice(model=self.model, transformer = self.transformer, pacmanEngine=self.pacmanEngine, threshold=self.threshold)

	def _getMDPActionAdviceInSubset(self, mdpActions: List[TMDPAction], mdpState: TMDPState, mdpOperations: TMDPOperations) -> List[TMDPAction]:
		"""!
		The Agent will receive an MDPOperations and
		must return a subset of a set of legal mdpActions
		"""
		if self.model == None:
			return mdpActions
		if len(mdpActions)==1:
			return mdpActions
		choices = []
		stateDescription = mdpOperations.stateDescription(mdpState)
		# print(self.pacmanEngine.printLayout(stateDescription)) # # debuging
		input_shape = self.model.layers[0].input_shape
		# Note I am using models supporting NHWC
		if input_shape[3] == 7: # input also contains food description
			x = np.expand_dims(self.pacmanEngine.getArrayWithFood(stateDescription),axis=0)
		elif input_shape[3] == 6: # input does not contain food description
			x = np.expand_dims(self.pacmanEngine.getArray(stateDescription),axis=0)
		else:
			raise Exception("Wrong input shape of model: "+str(input_shape))
		x = tf.transpose(x, [0, 2, 3, 1])
		actionValues = keras.backend.get_value(self.model(x))
		if self.transformer == None:
			actionValues = actionValues[0]
		else:
			actionValues = self.transformer.inverse_transform(actionValues)[0]
		# print(actionValues) # # debuging
		actionValues = actionValues - min(actionValues) # TODO: shift with a global value, not a local value
		maxValue = max(actionValues)
		# print(maxValue) # # debuging
		for action in mdpActions:
			# print(action.action)
			try:
				actionId = ['East','West','North','South'].index(action.action)
				# print(actionId)
			except:
				 raise Exception("multiple actions"+str([str(mdpAction) for mdpAction in mdpActions]))

			# print(actionValues[actionId],self.threshold*maxValue)
			if actionValues[actionId] >= self.threshold*maxValue:
				choices.append(action)
		if len(choices)==0:
			return mdpActions
		# print("here2",[action.action for action in choices])
		return choices

class MDPNNActionAdviceProgress( MDPActionAdviceInterface[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction] ):

	def __init__(self,modelSafety,modelProgress,transformerSafety,transformerProgress,pacmanEngine,threshold):
		self.modelSafety = modelSafety
		self.modelProgress = modelProgress
		self.transformerSafety = transformerSafety
		self.transformerProgress = transformerProgress
		self.pacmanEngine = pacmanEngine
		self.threshold = threshold
		if self.modelSafety is not None and self.modelProgress == None:
			self.mdpActionAdvice = MDPNNActionAdvice(modelSafety,transformerSafety,pacmanEngine, threshold)

	def deepCopy(self):
		return MDPNNActionAdvice(modelSafety=self.modelSafety, modelProgress=modelProgress, transformerSafety = self.transformerSafety, transformerProgress = self.transformerProgress, pacmanEngine=self.pacmanEngine, threshold=self.threshold)

	def _getMDPActionAdviceInSubset(self, mdpActions: List[TMDPAction], mdpState: TMDPState, mdpOperations: TMDPOperations) -> List[TMDPAction]:
		"""
		The Agent will receive an MDPOperations and
		must return a subset of a set of legal mdpActions
		"""
		if self.modelSafety == None and self.modelProgress == None:
			return mdpActions
		if self.modelSafety is not None and self.modelProgress == None:
			return self.mdpActionAdvice._getMDPActionAdviceInSubset(mdpActions,mdpState,mdpOperations)
		if len(mdpActions)==1:
			return mdpActions
		choices = []
		stateDescription = mdpOperations.stateDescription(mdpState)
		# print(self.pacmanEngine.printLayout(stateDescription)) # # debuging
		inputShapeSafety = self.modelSafety.layers[0].input_shape
		inputShapeProgress = self.modelProgress.layers[0].input_shape
		# Note I am using models supporting NHWC
		if inputShapeSafety[3] == 7: # input also contains food description
			xSafety = np.expand_dims(self.pacmanEngine.getArrayWithFood(stateDescription),axis=0)
			raise Exception("bad shape safety")
		elif inputShapeSafety[3] == 6: # input does not contain food description
			xSafety = np.expand_dims(self.pacmanEngine.getArray(stateDescription),axis=0)
		else:
			raise Exception("Wrong input shape of model: "+str(inputShapeSafety))
		if inputShapeProgress[3] == 7: # input also contains food description
			xProgress = np.expand_dims(self.pacmanEngine.getArrayWithFood(stateDescription),axis=0)
		elif inputShapeProgress[3] == 6: # input does not contain food description
			xProgress = np.expand_dims(self.pacmanEngine.getArray(stateDescription),axis=0)
			raise Exception("bad shape progress")
		else:
			raise Exception("Wrong input shape of model: "+str(inputShapeProgress))
		xSafety = tf.transpose(xSafety, [0, 2, 3, 1])
		xProgress = tf.transpose(xProgress, [0, 2, 3, 1])
		actionValuesSafety = keras.backend.get_value(self.modelSafety(xSafety))
		actionValuesProgress = keras.backend.get_value(self.modelProgress(xProgress))
		# print(actionValuesSafety)  ## debug
		# print(actionValuesProgress) ## debug
		if self.transformerSafety == None:
			actionValuesSafety = actionValuesSafety[0]
		else:
			actionValuesSafety = self.transformer.inverse_transform(actionValuesSafety)[0]
		if self.transformerProgress == None:
			actionValuesProgress = actionValuesProgress[0]
		else:
			actionValuesProgress = self.transformer.inverse_transform(actionValuesProgress)[0]
		actionValues = np.zeros(len(actionValuesProgress))
		for i in range(len(actionValuesSafety)):
			actionValues[i] = (actionValuesProgress[i]*40)+ 20 + (-230)*(1-actionValuesSafety[i])
		# print(actionValues) # # debuging
		# print("safety",actionValuesSafety)  ## debug
		# print("progress",actionValuesProgress) ## debug
		# print("total",actionValues,"\n") # # debug
		actionValues = actionValues - min(actionValues)
		maxValue = max(actionValues)
		# print(maxValue) # # debuging
		for action in mdpActions:
			# print(action.action)
			try:
				actionId = ['East','West','North','South'].index(action.action)
				# print(actionId)
			except:
				 raise Exception("multiple actions"+str([str(mdpAction) for mdpAction in mdpActions]))

			# print(actionValues[actionId],self.threshold*maxValue)
			if actionValues[actionId] >= self.threshold*maxValue:
				choices.append(action)
		if len(choices)==0:
			return mdpActions
		# print("here2",[action.action for action in choices])
		return choices

class MDPStormActionAdvice( MDPActionAdviceInterface[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction] ): ## TODO:
	"""! action advice that returns action over a given threshold according to Storm
	@params depth Unfording of the MDP with  horizon depth is created and Storm is used to find the probability of staying safe
	@params pacmanEngine
	@params threshold

	@returns choices set of actions
	"""

	def __init__(self,depth,pacmanEngine,threshold):
		self.depth = depth
		self.pacmanEngine = pacmanEngine
		self.threshold = threshold

	def _getMDPActionAdviceInSubset(self, mdpActions: List[TMDPAction], mdpState: TMDPState, mdpOperations: TMDPOperations) -> List[TMDPAction]:
		"""
		The Agent will receive an MDPOperations and
		must return a subset of a set of legal mdpActions
		"""
		if len(mdpActions)==1:
			return mdpActions
		choices = []
		stateDescription = mdpOperations.stateDescription(mdpState)
		# print(self.pacmanEngine.printLayout(stateDescription)) # # debuging
		prismFile = TEMP_DIR+os.sep+str(os.getpid())+'_advice.nm'
		height,width,numGhosts,layoutText,agentInfo = self.pacmanEngine.getInfo(stateDescription)
		actionValues = []
		for pacmanFirstAction in range(1,5):
			p = pacmanEngineNoFood(height,width,numGhosts,layoutText,agentInfo,pacmanFirstAction=pacmanFirstAction,drawDepth =self.depth, fname=prismFile)
			p.createPrismFile()
			value = getValue(prismFile,quiet=True)
			actionValues.append(value)
		os.remove(prismFile)
		maxValue = max(actionValues)
		# if maxValue == 1: # if there are always safe actions choose always safe actions
		# 	threshold = 1
		# else:
		# 	threshold = self.threshold
		for action in mdpActions:
			try:
				actionId = ['East','West','North','South'].index(action.action)
			except:
				 raise Exception("multiple actions"+str([str(mdpAction) for mdpAction in mdpActions]))

			if actionValues[actionId] >= self.threshold*maxValue:
				# print(actionName)
				choices.append(action)
		if len(choices)==0:
			return mdpActions
		return choices

# class MDPDebugActionAdvice( MDPActionAdviceInterface[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction] ): ## TODO:
#
# 	def __init__(self,advice1,advice2):
# 		self.advice1 = advice1
# 		self.advice2 = advice2
#
# 	def _getMDPActionAdviceInSubset(self, mdpActions: List[TMDPAction], mdpState: TMDPState, mdpOperations: TMDPOperations) -> List[TMDPAction]:
# 		stateDescription = mdpOperations.stateDescription(mdpState)
# 		# print(self.pacmanEngine.printLayout(stateDescription)) # # debuging
# 		mdpActions1 = self.advice1._getMDPActionAdviceInSubset(mdpActions, mdpState, mdpOperations)
# 		mdpActions2 = self.advice2._getMDPActionAdviceInSubset(mdpActions, mdpState, mdpOperations)
# 		s1 = set([action.action for action in mdpActions1])
# 		s2 = set([action.action for action in mdpActions2])
# 		for act in s1:
# 			if act not in s2:
# 				raise Exception("Actions "+act+" from "+str(s1)+" not in "+str(s2))
# 		for act in s2:
# 			if act not in s1:
# 				raise Exception("Actions "+act+" from "+str(s2)+" not in "+str(s1))
# 		# if s1 == s2:
# 		# 	raise Exception("Actions not equal "+str(s1)+" "+str(s2))
# 		return mdpActions1
#

class MDPNNActionStrategy( MDPProbabilisticActionStrategyInterface[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction] ):
	"""!
	A strategy that chooses a legal action uniformly at random from actions from Neural network.
	"""
	TMDPOperations = TypeVar("TMDPOperations",bound=MDPOperationsInterface[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction])

	def __init__(self, model, transformer, pacmanEngine, threshold) -> None:
		self.model = model
		self.transformer = transformer
		self.pacmanEngine = pacmanEngine
		self.threshold = threshold
		self.advice = MDPNNActionAdvice(model,transformer,pacmanEngine,threshold)

	def deepCopy(self) -> "MDPNNActionStrategy[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction]":
		return MDPNNActionStrategy(model=self.model, transformer = self.transformer, pacmanEngine=self.pacmanEngine, threshold=self.threshold)

	def _getDistribution(self, mdpActions: List[TMDPAction], mdpState: TMDPState, mdpOperations: TMDPOperations) -> util.ConsoleStrFloatCounter[TMDPAction]:

		mdpActions,mdpActionsFull = self.advice.getMDPActionAdviceInSubset(mdpActions, mdpState, mdpOperations, True)
		dist: util.ConsoleStrFloatCounter[TMDPAction] = util.ConsoleStrFloatCounter()
		for a in mdpActions: dist[a] = 1.0
		dist.normalize()
		return dist

class MDPUniformAdviceActionStrategy( MDPProbabilisticActionStrategyInterface[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction] ):
	"""!
	A strategy that chooses a legal action uniformly at random from legal actions.
	"""
	TMDPOperations = TypeVar("TMDPOperations",bound=MDPOperationsInterface[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction])

	def __init__(self, advice) -> None:
		self.advice = advice

	def deepCopy(self) -> "MDPNNActionStrategy[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction]":
		return MDPNNActionStrategy(advice=self.advice)

	def _getDistribution(self, mdpActions: List[TMDPAction], mdpState: TMDPState, mdpOperations: TMDPOperations) -> util.ConsoleStrFloatCounter[TMDPAction]:

		mdpActions,mdpActionsFull = self.advice.getMDPActionAdviceInSubset(mdpActions, mdpState, mdpOperations, True)
		dist: util.ConsoleStrFloatCounter[TMDPAction] = util.ConsoleStrFloatCounter()
		for a in mdpActions: dist[a] = 1.0
		dist.normalize()
		return dist

class MDPNNActionStrategyProgress( MDPProbabilisticActionStrategyInterface[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction] ):
	"""
	A strategy that chooses a legal action uniformly at random from actions from Neural network.
	"""
	TMDPOperations = TypeVar("TMDPOperations",bound=MDPOperationsInterface[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction])

	def __init__(self,modelSafety,modelProgress,transformerSafety,transformerProgress, pacmanEngine, threshold) -> None:
		self.modelSafety = modelSafety
		self.modelProgress = modelProgress
		self.transformerSafety = transformerSafety
		self.transformerProgress = transformerProgress
		self.pacmanEngine = pacmanEngine
		self.threshold = threshold
		self.advice = MDPNNActionAdviceProgress(modelSafety,modelProgress,transformerSafety,transformerProgress,pacmanEngine,threshold)

	def deepCopy(self) -> "MDPNNActionStrategyProgress[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction]":
		return MDPNNActionStrategyProgress(modelSafety=self.modelSafety, modelProgress = self.modelProgress, transformerSafety = self.transformerSafety, transformerProgress = self.transformerProgress, pacmanEngine=self.pacmanEngine, threshold=self.threshold)

	def _getDistribution(self, mdpActions: List[TMDPAction], mdpState: TMDPState, mdpOperations: TMDPOperations) -> util.ConsoleStrFloatCounter[TMDPAction]:

		mdpActions,mdpActionsFull = self.advice.getMDPActionAdviceInSubset(mdpActions, mdpState, mdpOperations, True)
		dist: util.ConsoleStrFloatCounter[TMDPAction] = util.ConsoleStrFloatCounter()
		for a in mdpActions: dist[a] = 1.0
		dist.normalize()
		return dist

# class MDPStormActionStrategy( MDPProbabilisticActionStrategyInterface[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction] ):
# 	"""
# 	A strategy that chooses a legal action uniformly at random from actions from Neural network.
# 	"""
# 	TMDPOperations = TypeVar("TMDPOperations",bound=MDPOperationsInterface[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction])
#
# 	def __init__(self, depth, pacmanEngine,threshold) -> None:
# 		self.depth = depth
# 		self.pacmanEngine = pacmanEngine
# 		self.threshold = threshold
# 		self.advice = MDPStormActionAdvice(depth,pacmanEngine,threshold)
#
# 	def deepCopy(self) -> "MDPStormActionStrategy[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction]":
# 		return MDPStormActionStrategy(depth=self.depth, pacmanEngine=self.pacmanEngine, threshold=self.threshold)
#
# 	def _getDistribution(self, mdpActions: List[TMDPAction], mdpState: TMDPState, mdpOperations: TMDPOperations) -> util.ConsoleStrFloatCounter[TMDPAction]:
#
# 		mdpActions,mdpActionsFull = self.advice.getMDPActionAdviceInSubset(mdpActions, mdpState, mdpOperations, True)
# 		dist: util.ConsoleStrFloatCounter[TMDPAction] = util.ConsoleStrFloatCounter()
# 		for a in mdpActions: dist[a] = 1.0
# 		dist.normalize()
# 		return dist

class MDPNNTraceEngine(Generic[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction]):
	TMDPOperations = TypeVar("TMDPOperations",bound=MDPOperationsInterface[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction])

	def getNNSimulationEngine(self, mdpState: TMDPState, mdpPredicates: List[TMDPPredicate], mdpOperations: TMDPOperations, horizonTrace: int, model, transformer, engine, threshold:float, quietTrace: bool, quietInfoStr: bool, printEachStepTrace: bool) -> MDPSimulationEngine[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction]:
		initState=mdpState.deepCopy()
		endState=mdpState.deepCopy()
		mdpPath: MDPPath[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction] = MDPPath(initState,[],[mdpPredicates])
		mdpExecutionEngine=MDPExecutionEngine(mdpOperations,MDPExecution(mdpPath,endState,0,False,1),0)
		mdpActionStrategy = MDPNNActionStrategy(model=model, transformer=transformer, pacmanEngine=engine, threshold=threshold)
		mdpActionTraceAdvice=MDPFullActionAdvice() # type: MDPFullActionAdvice[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction]
		mdpPathTraceAdvice=MDPFullPathAdvice() # type: Any
		mdpStateScoreTrace=MDPStateScoreZero() # type: Any
		optionsTraceEngine: OptionsSimulationEngine[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction] = OptionsSimulationEngine(horizon=horizonTrace, ignoreNonDecisionStates = False, mdpActionStrategy=mdpActionStrategy, mdpActionAdvice=mdpActionTraceAdvice, mdpPathAdvice=mdpPathTraceAdvice, mdpStateScore=mdpStateScoreTrace, alpha = 0.0, rejectFactor=1, quiet=quietTrace, quietInfoStr=quietInfoStr, printEachStep=printEachStepTrace, printCompact=True)
		mdpSimulationEngine = MDPSimulationEngine(mdpExecutionEngine, optionsTraceEngine)
		return mdpSimulationEngine

	def runNNTrace(self, numTraces: int, mdpState: TMDPState, mdpPredicates: List[TMDPPredicate], mdpOperations: TMDPOperations, horizonTrace: int, model, transformer, engine, threshold: float, quietTrace: bool, quietInfoStr: bool, printEachStepTrace: bool) -> List[Tuple[MDPExecutionEngine[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction], float, int]]:
		mdpSimulationEngine = self.getNNSimulationEngine(mdpState, mdpPredicates, mdpOperations, horizonTrace, model, transformer, engine, threshold,quietTrace,quietInfoStr,printEachStepTrace)
		results = mdpSimulationEngine.getSimulations(numTraces)
		return results

class MDPNNTraceEngineProgress(Generic[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction]):
	TMDPOperations = TypeVar("TMDPOperations",bound=MDPOperationsInterface[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction])

	def getNNSimulationEngine(self, mdpState: TMDPState, mdpPredicates: List[TMDPPredicate], mdpOperations: TMDPOperations, horizonTrace: int, modelSafety, modelProgress, transformerSafety, transformerProgress, engine, threshold:float, quietTrace: bool, quietInfoStr: bool, printEachStepTrace: bool) -> MDPSimulationEngine[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction]:
		initState=mdpState.deepCopy()
		endState=mdpState.deepCopy()
		mdpPath: MDPPath[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction] = MDPPath(initState,[],[mdpPredicates])
		mdpExecutionEngine=MDPExecutionEngine(mdpOperations,MDPExecution(mdpPath,endState,0,False,1),0)
		mdpActionStrategy = MDPNNActionStrategyProgress(modelSafety=modelSafety, modelProgress = modelProgress, transformerSafety=transformerSafety, transformerProgress=transformerProgress, pacmanEngine=engine, threshold=threshold)
		mdpActionTraceAdvice=MDPFullActionAdvice() # type: MDPFullActionAdvice[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction]
		mdpPathTraceAdvice=MDPFullPathAdvice() # type: Any
		mdpStateScoreTrace=MDPStateScoreZero() # type: Any
		optionsTraceEngine: OptionsSimulationEngine[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction] = OptionsSimulationEngine(horizon=horizonTrace, ignoreNonDecisionStates = False, mdpActionStrategy=mdpActionStrategy, mdpActionAdvice=mdpActionTraceAdvice, mdpPathAdvice=mdpPathTraceAdvice, mdpStateScore=mdpStateScoreTrace, alpha = 0.0, rejectFactor=1, quiet=quietTrace, quietInfoStr=quietInfoStr, printEachStep=printEachStepTrace, printCompact=True)
		mdpSimulationEngine = MDPSimulationEngine(mdpExecutionEngine, optionsTraceEngine)
		return mdpSimulationEngine

	def runNNTrace(self, numTraces: int, mdpState: TMDPState, mdpPredicates: List[TMDPPredicate], mdpOperations: TMDPOperations, horizonTrace: int, modelSafety, modelProgress, transformerSafety, transformerProgress, engine, threshold: float, quietTrace: bool, quietInfoStr: bool, printEachStepTrace: bool) -> List[Tuple[MDPExecutionEngine[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction], float, int]]:
		mdpSimulationEngine = self.getNNSimulationEngine(mdpState, mdpPredicates, mdpOperations, horizonTrace, modelSafety, modelProgress, transformerSafety, transformerProgress, engine, threshold, quietTrace, quietInfoStr, printEachStepTrace)
		results = mdpSimulationEngine.getSimulations(numTraces)
		return results

class MDPStormTraceEngine(Generic[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction]):
	TMDPOperations = TypeVar("TMDPOperations",bound=MDPOperationsInterface[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction])

	def getStormSimulationEngine(self, mdpState: TMDPState, mdpPredicates: List[TMDPPredicate], mdpOperations: TMDPOperations, horizonTrace: int, depth, engine, threshold:float, quietTrace: bool, quietInfoStr: bool, printEachStepTrace: bool) -> MDPSimulationEngine[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction]:
		initState=mdpState.deepCopy()
		endState=mdpState.deepCopy()
		mdpPath: MDPPath[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction] = MDPPath(initState,[],[mdpPredicates])
		mdpExecutionEngine=MDPExecutionEngine(mdpOperations,MDPExecution(mdpPath,endState,0,False,1),0)
		mdpStormAdvice=MDPStormActionAdvice(depth,engine,threshold)
		mdpActionStrategy = MDPUniformAdviceActionStrategy(mdpStormAdvice)
		mdpActionTraceAdvice=MDPFullActionAdvice() # type: MDPFullActionAdvice[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction]
		mdpPathTraceAdvice=MDPFullPathAdvice() # type: Any
		mdpStateScoreTrace=MDPStateScoreZero() # type: Any
		optionsTraceEngine: OptionsSimulationEngine[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction] = OptionsSimulationEngine(horizon=horizonTrace, ignoreNonDecisionStates = False, mdpActionStrategy=mdpActionStrategy, mdpActionAdvice=mdpActionTraceAdvice, mdpPathAdvice=mdpPathTraceAdvice, mdpStateScore=mdpStateScoreTrace, alpha = 0.0, rejectFactor=1, quiet=quietTrace, quietInfoStr=quietInfoStr, printEachStep=printEachStepTrace, printCompact=True)
		mdpSimulationEngine = MDPSimulationEngine(mdpExecutionEngine, optionsTraceEngine)
		return mdpSimulationEngine

	def runStormTrace(self, numTraces: int, mdpState: TMDPState, mdpPredicates: List[TMDPPredicate], mdpOperations: TMDPOperations, horizonTrace: int, depth, pacmanEngine, threshold: float, quietTrace: bool, quietInfoStr: bool, printEachStepTrace: bool) -> List[Tuple[MDPExecutionEngine[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction], float, int]]:
		mdpSimulationEngine = self.getStormSimulationEngine(mdpState, mdpPredicates, mdpOperations, horizonTrace, depth, pacmanEngine,threshold,quietTrace,quietInfoStr,printEachStepTrace)
		results = mdpSimulationEngine.getSimulations(numTraces)
		return results

class MDPUniformAdviceTraceEngine(Generic[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction]):
	TMDPOperations = TypeVar("TMDPOperations",bound=MDPOperationsInterface[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction])

	def getSimulationEngine(self, mdpState: TMDPState, mdpPredicates: List[TMDPPredicate], mdpOperations: TMDPOperations, horizonTrace: int, uniformAdvice, quietTrace: bool, quietInfoStr: bool, printEachStepTrace: bool) -> MDPSimulationEngine[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction]:
		initState=mdpState.deepCopy()
		endState=mdpState.deepCopy()
		mdpPath: MDPPath[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction] = MDPPath(initState,[],[mdpPredicates])
		mdpExecutionEngine=MDPExecutionEngine(mdpOperations,MDPExecution(mdpPath,endState,0,False,1),0)
		mdpActionStrategy = MDPUniformAdviceActionStrategy(uniformAdvice)
		mdpActionTraceAdvice=MDPFullActionAdvice() # type: MDPFullActionAdvice[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction]
		mdpPathTraceAdvice=MDPFullPathAdvice() # type: Any
		mdpStateScoreTrace=MDPStateScoreZero() # type: Any
		optionsTraceEngine: OptionsSimulationEngine[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction] = OptionsSimulationEngine(horizon=horizonTrace, ignoreNonDecisionStates = False, mdpActionStrategy=mdpActionStrategy, mdpActionAdvice=mdpActionTraceAdvice, mdpPathAdvice=mdpPathTraceAdvice, mdpStateScore=mdpStateScoreTrace, alpha = 0.0, rejectFactor=1, quiet=quietTrace, quietInfoStr=quietInfoStr, printEachStep=printEachStepTrace, printCompact=True)
		mdpSimulationEngine = MDPSimulationEngine(mdpExecutionEngine, optionsTraceEngine)
		return mdpSimulationEngine

	def runTrace(self, numTraces: int, mdpState: TMDPState, mdpPredicates: List[TMDPPredicate], mdpOperations: TMDPOperations, horizonTrace: int, uniformAdvice, quietTrace: bool, quietInfoStr: bool, printEachStepTrace: bool) -> List[Tuple[MDPExecutionEngine[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction], float, int]]:
		mdpSimulationEngine = self.getSimulationEngine(mdpState, mdpPredicates, mdpOperations, horizonTrace, uniformAdvice,quietTrace,quietInfoStr,printEachStepTrace)
		results = mdpSimulationEngine.getSimulations(numTraces)
		return results

if __name__ == '__main__':
	# import math

	# from stormMdpClasses import prismToModel
	# from traceEngine import runTraces
	X,Y,numGhosts,layoutText,agentInfo = readFromFile(sys.argv[1])#'layouts/pacman/halfClassic.lay'
	prismFile = 'prismFiles/pacmanTest.sm'
	p = pacmanEngine(X,Y,numGhosts,layoutText,agentInfo,prismFile)
	p.createPrismFile()
	p.savePickle()
#    d = {"capsule0_0":True,"f10_10":1,"f10_2":1,"f10_4":0,"f10_6":0,"f10_8":0,"f12_10":1,"f12_2":1,"f12_4":0,"f12_6":0,"f12_8":0,"f14_10":1,"f14_2":1,"f14_4":1,"f14_6":0,"f14_8":0,"f2_10":0,"f2_2":1,"f2_4":1,"f2_6":0,"f2_8":0,"f4_10":0,"f4_2":0,"f4_4":0,"f4_6":2,"f4_8":0,"f6_10":0,"f6_2":0,"f6_4":0,"f6_6":0,"f6_8":0,"f8_10":0,"f8_2":1,"f8_4":1,"f8_6":0,"f8_8":0,"foodEaten":0,"steps":0,"timer0":29,"token":1,"xg0":5,"xp":6,"yg0":10,"yp":6}

#    p.printLayout(d)
#    path = 'pacmanTest.sm'
	# model = prismToModel(prismFile)
	#
	# runTraces(prismFile, 1, 20, 100, 10, 4, math.sqrt(2)/2, 0, 100, True, False,p.printLayout)
	# f = open('dot.dot','w+')
	# s = model.to_dot()
	# print(s,file=f)
	# f.close()
