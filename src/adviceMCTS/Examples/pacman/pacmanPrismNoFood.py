import sys, json, os
import numpy as np
from tensorflow import keras

# CURRENTPWD = os.path.dirname(os.path.abspath(__file__))
# SRC_DIR = os.path.join(CURRENTPWD, '../src')
# sys.path.append(SRC_DIR)
#
from adviceMCTS.Examples.simulationClasses import *
import adviceMCTS.util as util

WORK_DIR = os.environ.get('GLOBALSCRATCH')
if WORK_DIR == None:
	WORK_DIR = ""
else:
	WORK_DIR = WORK_DIR + os.sep

TEMP_DIR = WORK_DIR+'tempFiles'+os.sep+'prism'
util.mkdir(TEMP_DIR)

def readFromFile(fileName):
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
	# print(X,Y,numGhosts,layoutText,agentInfo)
	f.close()
	return (X,Y,numGhosts,layoutText,agentInfo)

class Layout:
	"""
	A Layout manages the static information about the game board.
	"""
	# X Y numGhosts
	# Foods and walls in a layout file with height X and width Y :
	# (0,0)     (0,1)   .  .  .  (0,Y-1)
	# (1,0)     (1,1)               .
	#  .                .           .
	#  .                  .         .
	#  .                    .       .
	# (X-1,0)   (X-1,1)  . . . (X-1,Y-1)
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
		self.pacmanPositions.append(agentInfo[0][0])
		for g in agentInfo[1:]:
			self.ghostPositions.append(g[0])
			self.ghostDirection.append(g[1])
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

	def getArray(self):
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

class pacmanEngine:

	def __init__(self,X,Y,numGhosts,layoutText,agentInfo, drawDepth, pacmanFirstAction=0, fname=None):  #drawDepth, timermax, scaredFactor,

		self.layout = Layout(X,Y,numGhosts,layoutText,agentInfo) # in layout row and columns are counted from bottom left corner
		self.fname = fname
		self.width = self.layout.width
		self.height = self.layout.height
		self.walls = self.layout.walls
		self.food = self.layout.food
		self.drawDepth = drawDepth
		self.pacmanPosition = self.layout.pacmanPositions[0] # Assuming 1 pacman
		self.ghostPositions = self.layout.ghostPositions # of form [(xg,yg),...]
		self.numGhosts = self.layout.numGhosts
		self.ghostDirection = self.layout.ghostDirection
		self.pacmanFirstAction = pacmanFirstAction # [0 = not specified, 1 = east, 2 = west, 3 = north, 4 = south]
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
	#     return (horizon*(2+self.numGhosts))

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

	# def _totalFood(self):
	#     totalFood = 'formula totalFood ='
	#     for i in range(self.height): # i is counted from top to bottom of grid starting from 0
	#         for j in range(self.width): # j is counted from left to right of grid starting from 0
	#             totalFood+=f' f{i}_{j} +'
	#     totalFood = totalFood[:-1]+';'
	#     print(totalFood, file=self.out)

	def _initialize(self):
		print('mdp\n',file=self.out)
		for i in range(self.height): # # i is counted from top to bottom of grid starting from 0
			for j in range(self.width): # j is counted from left to right of grid starting from 0
				# constants for co-ordinates
				print(f'const int x{i}_{j} = {i};',file=self.out) # ints for every x-position in the grid
				print(f'const int y{i}_{j} = {j};',file=self.out) # ints for every y-position in the grid
				# walls
				print(f'const int w{i}_{j} = {[0,1][self.walls[i][j]]};',file=self.out) # ints for walls
		# self._totalFood()
		# print('const int drawDepth = %s;' %self.drawDepth,file=self.out)
		isLossString = 'formula isLoss = (token = 0) & ('
		for g in range(self.numGhosts):
			self._ghostDirections(g)
			isLossString+=f'isLoss{g} | '
		isLossString = isLossString[:-3]+');'
		print(isLossString,file=self.out)
		# print('formula isWin = (token = 0) & (totalFood = 0);',file=self.out)
		print(f'formula isDraw = (token = 0) &  steps = {self.drawDepth} & !isLoss;',file=self.out)

	def _moduleArbiter(self):
		print('\nmodule arbiter\n',file=self.out)
		print(f'token : [0 .. {2+self.numGhosts}] init 0;',file=self.out)
		'''
		1 token for pacman
		1 token for each ghosts
		1 token to update food and check at the end
		1 extra token for the win/loss state
		'''

		print('result : [0 .. 3] init 0;',file=self.out)
		'''
		1 = Win
		2 = Loss
		3 = Draw
		'''
		print(f'pacmanFirstAction : [0 .. 4] init {self.pacmanFirstAction};',file=self.out)
		'''
		0  init
		1  east
		2  west
		3  north
		4  south
		'''

		# token = 0

		# actions for pacman
		print('',file=self.out)
		print('[East] (!(isDraw | isLoss) & token = 0 & pacmanFirstAction = 1 & steps = 0) -> 1: (token\' = 1);',file=self.out)
		print('[West] (!(isDraw | isLoss) & token = 0 & pacmanFirstAction = 2 & steps = 0) -> 1: (token\' = 1);',file=self.out)
		print('[North] (!(isDraw | isLoss) & token = 0 & pacmanFirstAction = 3 & steps = 0) -> 1: (token\' = 1);',file=self.out)
		print('[South] (!(isDraw | isLoss) & token = 0 & pacmanFirstAction = 4 & steps = 0) -> 1: (token\' = 1);',file=self.out)
		print('[East] (!(isDraw | isLoss) & token = 0 & (!(steps = 0) | pacmanFirstAction = 0)) -> 1: (token\' = 1);',file=self.out)
		print('[West] (!(isDraw | isLoss) & token = 0 & (!(steps = 0) | pacmanFirstAction = 0)) -> 1: (token\' = 1);',file=self.out)
		print('[North] (!(isDraw | isLoss) & token = 0 & (!(steps = 0) | pacmanFirstAction = 0)) -> 1: (token\' = 1);',file=self.out)
		print('[South] (!(isDraw | isLoss) & token = 0 & (!(steps = 0) | pacmanFirstAction = 0)) -> 1: (token\' = 1);',file=self.out)

		# token = 1 ... numGhosts
		# actions for ghosts
		print('',file=self.out)
		for i in range (self.numGhosts):
			print(f'[g{i}] (token = {i+1}) -> 1: (token\' = {i+2});' ,file=self.out)

		# token = numGhosts + 1
		# update food
		print(f'\n[update] (token = {self.numGhosts+1}) -> 1: (token\' = 0);\n',file=self.out)

		# print(f'[] (isWin & token = 0) -> 1: (token\' = {self.numGhosts+2}) & (result\' = 1);',file=self.out)
		print(f'[] (isLoss) -> 1: (token\' = {self.numGhosts+2}) & (result\' = 2);',file=self.out)
		print(f'[] (isDraw) -> 1: (token\' = {self.numGhosts+2}) & (result\' = 3);',file=self.out)
		print(f'[] (token = {self.numGhosts+2}) -> 1: (token\' = token);',file=self.out)

#        print('',file=self.out)
#        print('[] ((isDraw | isWin | isLoss) & token = 0) -> 1: (token\' = 0);',file=self.out)
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

	# def _initialFood(self):
	#     for i in range(self.height): # i is counted from left to right of grid starting from 0
	#         for j in range(self.width): # j is counted from bottom to top of grid starting from 0
	#             print(f'f{i}_{j} : [0 .. 1] init {[0,1][self.food[i][j]]};',file=self.out) # ints for food

	def _moduleUpdate(self):
		print('\nmodule check\n',file=self.out)
		print(f'steps: [0 .. {self.drawDepth}] init 0;\n',file=self.out)
		# print('foodEaten: [0 .. 1] init 0;',file=self.out) # update if a food or capsule is eaten in a step

		# self._initialFood()

		# for i in range(1,self.height-1):
			# for j in range(1,self.width-1): # assuming there are walls in the border
				# eat food
				# print(f'[update] (xp{i}_{j}=xp & yp{i}_{j}=yp) -> 1: (f{i}_{j}\'=0) & (foodEaten\'=f{i}_{j});',file=self.out)
		print('[update] true -> 1: (steps\'=steps+1);' ,file=self.out)
		print('endmodule\n',file=self.out)

	def _labels(self):
		# print('\nlabel "Win" = (result = 1);',file=self.out)
		print('label "Loss" = (result = 2);',file=self.out)
		print('label "Draw" = (result = 3);',file=self.out)
		pass

	def _rewards(self):
		print('\nrewards',file=self.out)
		# -1 for each move
		# print('[East] true: -1;',file=self.out)
		# print('[West] true: -1;',file=self.out)
		# print('[North] true: -1;',file=self.out)
		# print('[South] true: -1;',file=self.out)

		# reward for eating food
		# print('(foodEaten = 1 & token = 0) : 1;' ,file=self.out)
		# print('(foodEaten = 2 & token = 0): 10;' ,file=self.out)
		# for i in range (self.numGhosts):
			# print('(eaten%s = 1 & token = %s): 10;' %(i,3*i+2),file=self.out)
		# reward for eating all food
		# print('isWin: 500;',file=self.out)

		# penalty for dying
		# print('isLoss: -500;',file=self.out)

		print('isDraw: 1;',file=self.out)

		print('endrewards',file=self.out)

	def createPrismFile(self):
		self._openOutput()
		self._initialize()
		self._moduleArbiter()
		self._modulePacman()
		for g in range(self.numGhosts):
			self._moduleGhost(g)
		self._moduleUpdate()
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
			currentLayout = [[False for j in range(self.width)] for i in range(self.height)]
			for i in range(self.height):
				for j in range(self.width):
					if self.walls[i][j]:
						currentLayout[i][j] = '%'
					else:
						foodKey = 'f'+str(i)+'_'+str(j)
						if foodKey not in d:
							currentLayout[i][j] = ' '
						elif d[foodKey] == 1:
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
			# results = s[2:]
			# if 'Win' in results:
			#     r = 'Win'
			# elif 'Loss' in results:
			#     r = 'Loss'
			# elif 'Draw' in results:
			#     r = 'Draw'
			# elif 'deadlock' in results:
			#     r = 'Deadlock'
			# else:
			#     r = ''
			s = ''
			for line in currentLayout:
				for l in line:
					s+=str(l)
				s+='\n'
			return s

	def getArray(self,d=None):
		if d == None:
			return self.layout.getArray()
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
				# else:
				# 	foodKey = 'f'+str(i)+'_'+str(j)
				# 	if d[foodKey] == 1:
				# 		line += '.'
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

def getLayoutFromArray(a):
	size = a.shape
	X = size[2]
	Y = size[3]
	layout = [[' ' for j in range(Y)] for i in range(X)]
	for i in range(X):
		for j in range(Y):
			if a[0][0][i][j] == 1:
				layout[i][j] = '%'
	s = "\n".join(["".join(layout[i]) for i in range(X)])
	for k in range(1,size[1]):
		for i in range(X):
			for j in range(Y):
				if a[0][k][i][j] == 1:
					s+="\n"+str(i)+" "+str(j)+" "+str(k-1)
	s+="\n"
	return s

def createEngine(fname,drawDepth):
	X,Y,numGhosts,layoutText,agentInfo = readFromFile(fname)
	prismFile = TEMP_DIR+os.sep+fname.split(os.sep)[-1][:-4]+'_'+str(os.getpid())+'.nm'
	p = pacmanEngine(X,Y,numGhosts,layoutText,agentInfo,drawDepth, pacmanFirstAction=0, fname=prismFile)
	return(p)

# class MDPNNActionAdvice( MDPActionAdviceInterface[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction] ):
#
# 	def __init__(self,model,transformer,pacmanEngine,threshold):
# 		self.model = model
# 		self.transformer = transformer
# 		self.pacmanEngine = pacmanEngine
# 		self.threshold = threshold
#
# 	def deepCopy(self):
# 		return MDPNNActionAdvice(model=self.model, transformer = self.transformer, pacmanEngine=self.pacmanEngine, threshold=self.threshold)
#
# 	def _getMDPActionAdviceInSubset(self, mdpActions: List[TMDPAction], mdpState: TMDPState, mdpOperations: TMDPOperations) -> List[TMDPAction]:
# 		"""
# 		The Agent will receive an MDPOperations and
# 		must return a subset of a set of legal mdpActions
# 		"""
# 		if self.model == None:
# 			return mdpActions
# 		if len(mdpActions)==1:
# 			return mdpActions
# 		choices = []
# 		stateDescription = mdpOperations.stateDescription(mdpState)
# 		input_shape = self.model.layers[0].input_shape
# 		# Note I am using models supporting NHWC
# 		if input_shape[3] == 7: # input also contains food description
# 			x = np.expand_dims(self.pacmanEngine.getArrayWithFood(stateDescription),axis=0)
# 		elif input_shape[3] == 6: # input does not contain food description
# 			x = np.expand_dims(self.pacmanEngine.getArray(stateDescription),axis=0)
# 		else:
# 			raise Exception("Wrong input shape of model: "+str(input_shape))
# 		x = tf.transpose(x, [0, 2, 3, 1])
# 		actionValues = keras.backend.get_value(self.model(x))[0]
# 		if self.transformer == None:
# 			pass
# 		else:
# 			actionValues = self.transformer.inverse_transform(actionValues)
# 		# print(actionValues)
# 		maxValue = max(actionValues)
# 		for action in mdpActions:
# 			try:
# 				actionId = ['East','West','North','South'].index(action.action)
# 			except:
# 				 raise Exception("multiple actions"+str([str(mdpAction) for mdpAction in mdpActions]))
#
# 			if actionValues[actionId] >= self.threshold*maxValue:
# 				choices.append(action)
# 		if len(choices)==0:
# 			return mdpActions
# 		return choices


if __name__ == '__main__':
	# import math

	# from stormMdpClasses import prismToModel
	# from traceEngine import runTraces
	X,Y,numGhosts,layoutText,agentInfo = readFromFile(sys.argv[1])#'layouts/pacman/halfClassic.lay'
	prismFile = 'prismFiles/pacmanTest.sm'
	p = pacmanEngine(X,Y,numGhosts,layoutText,agentInfo,prismFile)
	p.createPrismFile()
	# p.savePickle()
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
