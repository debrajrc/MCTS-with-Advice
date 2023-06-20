import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from simulationClasses import MDPPathAdviceInterface, MDPActionAdviceInterface, MDPStateScoreInterface, MDPExecutionEngine
from stormMdpClasses import MDPState, MDPOperations, MDPAction

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models

import stormpy, util

dirname = os.path.dirname(__file__)

sys.path.append(dirname)
from pacmanPrism import PacmanEngine

# terminal reward
class MDPStateScore(MDPStateScoreInterface):
	def getScore(self, executionEngine : MDPExecutionEngine) -> float:
		return 0

# selection advice
class MDPSafeActionAdvice( MDPActionAdviceInterface):

	def getMDPActionAdvice(self, mdpState : MDPState, mdpOperations: MDPOperations, quietInfoStr: bool) -> tuple[list[MDPAction]]:
		choices = mdpOperations.getLegalActions(mdpState)
		if not quietInfoStr:
			for mdpAction in choices:
				if mdpAction.infoStr != '':
					mdpAction.infoStr += '#'
				mdpAction.infoStr += 'AdviceFull'
		return choices, choices

# path advice
class MDPNonLossPathAdvice(MDPPathAdviceInterface):

	def isValidPath(self, mdpExecutionEngine : MDPExecutionEngine) -> bool:
		mdpPredicatesList = mdpExecutionEngine.mdpOperations.getPredicates(mdpExecutionEngine.mdpEndState())
		for predicate in mdpPredicatesList:
			if predicate.name == "Loss":
				return False
		return True
	
# a function to print the states nice
def niceStr(stateDict : dict) -> str:
	p = PacmanEngine.fromFile(dirname+"/layouts/halfClassic.lay")
	return p.printLayout(stateDict)

class MDPStateScoreDistanceOld(MDPStateScoreInterface):
	"""! Distance function that uses a linear combination of distance (exact distance in the graph) from ghosts and remaining foods
	"""
	def __init__(self):
		p = PacmanEngine.fromFile(dirname+"/layouts/halfClassic.lay")
		(X,Y,numGhosts,walls) = p.getLayoutDescription()
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

class MDPNNActionAdvice( MDPActionAdviceInterface):
	"""! action advice that returns action over a given threshold
	@params model NN model
	@params transformer a sklearn.preprocessing.QuantileTransformer object
	@params pacmanEngine
	@params threshold

	@returns choices set of actions
	"""
	def __init__(self):
		pacmanEngine = PacmanEngine.fromFile(dirname+"/layouts/halfClassic.lay")
		modelFile = dirname+"/models/all.h5"
		model = models.load_model(modelFile, compile=False)
		self.model = model
		self.pacmanEngine = pacmanEngine
		self.threshold = 0.9

	def deepCopy(self):
		return MDPNNActionAdvice(model=self.model, pacmanEngine=self.pacmanEngine, threshold=self.threshold)

	def _getMDPActionAdviceInSubset(self, mdpActions, mdpState, mdpOperations):
		"""!
		The Agent will receive an MDPOperations and
		must return a subset of a set of legal mdpActions
		"""
		if self.model == None:
			return mdpActions
		if len(mdpActions)==1:
			return mdpActions
		choices = []
		# print("\nAvailable actions:", [action.action for action in mdpActions]) # # debuging
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
		actionValues = actionValues[0]
		# print(actionValues) # # debuging
		maxValue = max(actionValues)
		# print(maxValue) # # debuging
		for action in mdpActions:
			# print(action.action)
			try:
				actionId = ['East','West','North','South'].index(action.action)
				# print(actionId)
			except:
				raise Exception("multiple actions"+str([str(mdpAction) for mdpAction in mdpActions]))
			pass 
			if actionValues[actionId] >= self.threshold*maxValue:
				choices.append(action)
		if len(choices)==0:
			return mdpActions
		# print("Best actions:", [action.action for action in choices], "\n") # # debuging
		return choices
	
class MDPStormActionAdvice( MDPActionAdviceInterface ):
	"""! action advice that returns action over a given threshold according to Storm
	@params depth Unfording of the MDP with  horizon depth is created and Storm is used to find the probability of staying safe
	@params pacmanEngine
	@params threshold

	@returns choices set of actions
	"""

	def __init__(self):
		depth = 3
		self.depth = depth
		pacmanEngine = PacmanEngine.fromFile(dirname+"/layouts/halfClassic.lay")
		self.pacmanEngine = pacmanEngine
		self.threshold = 0.9

	def _getMDPActionAdviceInSubset(self, mdpActions, mdpState, mdpOperations):
		"""
		The Agent will receive an MDPOperations and
		must return a subset of a set of legal mdpActions
		"""
		if len(mdpActions)==1:
			return mdpActions
		choices = []
		# print("\nAvailable actions:", [action.action for action in mdpActions]) # # debuging
		stateDescription = mdpOperations.stateDescription(mdpState)
		# print(self.pacmanEngine.printLayout(stateDescription)) # # debuging
		height,width,numGhosts,layoutText,agentInfo = self.pacmanEngine.getInfo(stateDescription)
		actionValues = []
		for pacmanFirstAction in range(1,5):
			util.mkdir(dirname+"/temp")
			prismFile = dirname+"/temp/"+str(os.getpid())+'_advice.nm'
			p = self.pacmanEngine.newEngine(agentInfo, prismFile, self.depth, pacmanFirstAction)
			# p = pacmanEngineNoFood(height,width,numGhosts,layoutText,agentInfo,pacmanFirstAction=pacmanFirstAction,drawDepth =self.depth, fname=prismFile)
			p.createPrismFile(noFood=True)
			prism_program = stormpy.parse_prism_program(prismFile)
			formula_str = "Pmax=? [(G ! isLoss)]"
			properties = stormpy.parse_properties(formula_str, prism_program)
			model = stormpy.build_model(prism_program, properties)
			# print(model)
			initial_state = model.states[0]
			if 'deadlock' in list(model.labels_state(initial_state)):
				value = 0 #(float('nan'))
			else:
				result = stormpy.model_checking(model, properties[0],only_initial_states=True)
				value = result.at(initial_state)
			actionValues.append(value)
			os.remove(prismFile)
			# print("action",pacmanFirstAction,"value",value) # # debuging
		# print(actionValues) # # debuging
		maxValue = max(actionValues)
		if maxValue == 1: # if there are always safe actions choose always safe actions
			threshold = 1
		else:
			threshold = self.threshold
		for action in mdpActions:
			try:
				actionId = ['East','West','North','South'].index(action.action)
			except:
				raise Exception("multiple actions"+str([str(mdpAction) for mdpAction in mdpActions]))
			if actionValues[actionId] >= threshold*maxValue:
				# print(actionName)
				choices.append(action)
		if len(choices)==0:
			return mdpActions
		# print("Best actions:", [action.action for action in choices], "\n") # # debuging
		return choices

