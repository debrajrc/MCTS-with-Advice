import sys, json, os
import numpy as np

def readFromFile(fileName):
	"""! Given a layout file, extracts information from it

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

	def processLayoutChar(self, x, y, layoutChar):
		if layoutChar == '%':
			self.walls[x][y] = True
		elif layoutChar == '.':
			self.food[x][y] = True
			self.totalFood += 1

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
			array[self.ghostDirection[g]+1][self.ghostPositions[g][0]][self.ghostPositions[g][1]] = 1
		return(array)

##
# class to create a prism file for pacman and do operations
class PacmanEngine:

	def __init__(self,X,Y,numGhosts,layoutText,agentInfo,fname=None,drawDepth=None,pacmanFirstAction=0): 
		self.layout = Layout(X,Y,numGhosts,layoutText,agentInfo) # in layout row and columns are counted from bottom left corner
		self.fname = fname
		self.width = self.layout.width
		self.height = self.layout.height
		self.walls = self.layout.walls
		self.food = self.layout.food
		self.pacmanPosition = self.layout.pacmanPositions[0] # Assuming 1 pacman
		self.ghostPositions = self.layout.ghostPositions # of form [(xg,yg),...]
		self.ghostDirection = self.layout.ghostDirection
		self.numGhosts = self.layout.numGhosts
		self.pacmanFirstAction = pacmanFirstAction
		self.drawDepth = drawDepth

	def __str__(self):
		return (str(self.layout))

	@classmethod
	def fromFile(cls,fileName, fname=None):
		(X,Y,numGhosts,layoutText,agentInfo) = readFromFile(fileName)
		return cls(X,Y,numGhosts,layoutText,agentInfo, fname)

	def _openOutput(self):
		if self.fname == None:
			self.out = sys.stdout
		else:
			file = open(self.fname,'w+')
			self.out = file

	def newEngine(self, agentInfo, fname, drawDepth, pacmanFirstAction=0):
		pacmanEngine = PacmanEngine(self.height,self.width,len(agentInfo) -1 , self.layout.layoutText, agentInfo, fname, drawDepth,pacmanFirstAction)
		return pacmanEngine
	
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

	def _initialize(self, noFood=False):
		print('mdp\n',file=self.out)
		for i in range(self.height): # # i is counted from top to bottom of grid starting from 0
			for j in range(self.width): # j is counted from left to right of grid starting from 0
				# constants for co-ordinates
				print(f'const int x{i}_{j} = {i};',file=self.out) # ints for every x-position in the grid
				print(f'const int y{i}_{j} = {j};',file=self.out) # ints for every y-position in the grid
				# walls
				print(f'const int w{i}_{j} = {[0,1][self.walls[i][j]]};',file=self.out) # ints for walls
		if not noFood:
			self._totalFood()
		# print('const int drawDepth = %s;' %self.drawDepth,file=self.out)
		isLossString = f'formula isLoss = (token = 0) & ('
		for g in range(self.numGhosts):
			self._ghostDirections(g)
			isLossString+=f'isLoss{g} | '
		isLossString = isLossString[:-3]+');'
		print(isLossString,file=self.out)
		if not noFood:
			print(f'formula isWin = (token = 0) & (totalFood = 0);',file=self.out)
		else:
			print(f'formula isDraw = (token = 0) &  (steps = {self.drawDepth}) & !isLoss;',file=self.out)
	
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

	def _moduleArbiterNoFood(self):
		print('\nmodule arbiter\n',file=self.out)
		print(f'token : [0 .. {2+self.numGhosts}] init 0;',file=self.out)
		'''
		1 token for pacman
		1 token for each ghosts
		1 token to update the step counter
		1 extra token for the win/loss state
		'''

		print('result : [0 .. 3] init 0;',file=self.out)
		'''
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
		# print(f'steps: [0 .. 1] init 0;\n',file=self.out)

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
		# update
		print(f'\n[update] (token = {self.numGhosts+1}) -> 1: (token\' = 0);\n',file=self.out)

		# print(f'[] (isWin & token = 0) -> 1: (token\' = {self.numGhosts+2}) & (result\' = 1);',file=self.out)
		print(f'[] (isLoss) -> 1: (token\' = {self.numGhosts+2}) & (result\' = 2);',file=self.out)
		print(f'[] (isDraw) -> 1: (token\' = {self.numGhosts+2}) & (result\' = 3);',file=self.out)
		print(f'[] (token = {self.numGhosts+2}) -> 1: (token\' = token);',file=self.out)

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
		print('endmodule\n',file=self.out)

	def _moduleUpdate(self): # this is for the case when there is no food
		print('\nmodule check\n',file=self.out)
		print(f'steps: [0 .. {self.drawDepth}] init 0;\n',file=self.out)
		print('[update] true -> 1: (steps\'=steps+1);' ,file=self.out)
		print('endmodule\n',file=self.out)

	def _labels(self, noFood=False):
		if not noFood:
			print('\nlabel "Win" = (result = 1);',file=self.out)
		else:
			print('label "Draw" = (result = 3);',file=self.out)
		print('label "Loss" = (result = 2);',file=self.out)

	def _rewards(self, noFood=False):
		print('\nrewards',file=self.out)
		if not noFood:
			# -1 for each move
			print('[East] true : -1;',file=self.out)
			print('[West] true: -1;',file=self.out)
			print('[North] true: -1;',file=self.out)
			print('[South] true: -1;',file=self.out)

			# reward for eating food
			print('(foodEaten = 1 & token = 0) : 10;' ,file=self.out)

			# reward for eating all food
			print('isWin: 500;',file=self.out)

			# penalty for dying
			print('isLoss: -500;',file=self.out)
		else:
			print('isDraw: 1;',file=self.out)
		print('endrewards',file=self.out)

	def createPrismFile(self, noFood=False):
		self._openOutput()
		self._initialize(noFood)
		if not noFood:
			self._moduleArbiter()
		else:
			self._moduleArbiterNoFood()
		self._modulePacman()
		for g in range(self.numGhosts):
			self._moduleGhost(g)
		if not noFood:
			self._moduleUpdateFood()
		else:
			self._moduleUpdate()
		self._labels(noFood)
		self._rewards(noFood)
		if self.out != sys.stdout:
			self.out.close()
		return(self.fname)

	def savePickle(self):
		import pickle
		try:
			out = p.__dict__.pop('out')
		except:
			pass
		pickle.dump(p, open(str(self.fname[:-3])+'.dump', 'wb'))
		try:
			p.out = out
		except:
			pass

	def printLayout(self,d): # prints current state of the game
			s = ''
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
		for ghost in range(self.numGhosts):
			(xg,yg) = d['xg'+str(ghost)], d['yg'+str(ghost)]
			distance=abs(xg-xp)+abs(yg-yp)# distance between ghost number ghost and pacman
			ghostValuation += -1/(1+distance)
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
	prismFile = fname.split(os.sep)[-1][:-4]+'_'+str(os.getpid())+'.nm' #TEMP_DIR+os.sep+
	p = PacmanEngine(X,Y,numGhosts,layoutText,agentInfo,prismFile)
	return(p)

if __name__ == '__main__':
	input = sys.argv
	X,Y,numGhosts,layoutText,agentInfo = readFromFile(sys.argv[1])#'layouts/halfClassic.lay'
	prismFile = 'pacman.nm'
	p = PacmanEngine(X,Y,numGhosts,layoutText,agentInfo,prismFile,None,0)
	p.createPrismFile(noFood = False)
	# p.savePickle()
