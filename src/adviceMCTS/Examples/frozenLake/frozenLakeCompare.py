import glob, os, sys, keras, gc
import numpy as np
import tensorflow as tf
import memory_profiler as mp

CURRENTPWD = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(CURRENTPWD, '..'+os.sep+'src'))

from adviceMCTS.mdpClasses import *
from adviceMCTS.Examples.frozenLake.frozenLakeMdpClasses import *
from adviceMCTS.simulationClasses import *
from adviceMCTS.Examples.frozenLake.frozenLake import *
from adviceMCTS.Examples.frozenLake.frozenLakeStorm import *

WORK_DIR = os.environ.get('GLOBALSCRATCH')
if WORK_DIR == None:
	WORK_DIR = ""
else:
	WORK_DIR = WORK_DIR + os.sep

LAYOUT_SET = 'NN10x10_Compare'
LAYOUTS_DIR = WORK_DIR+'layouts'+os.sep+'frozenLake'+os.sep+LAYOUT_SET
MODEL_DIR = 'frozenLake'+os.sep+'models'+os.sep+"NN10x10"

def getValueFromLayoutMCTS(layout, horizon, numMCTSIters, numSims, horizonTrace, stormAtRoot = False): # runs mcts and reports average wins
	if stormAtRoot:
		mdpActionAdviceRoot=MDPStormActionAdvice(threshold = 0.99)
	else:
		mdpActionAdviceRoot = MDPFullActionAdvice()
	optionsSimulationEngine=OptionsSimulationEngine(horizon=horizon, ignoreNonDecisionStates = True, mdpActionStrategy=MDPUniformActionStrategy(), mdpActionAdvice=MDPFullActionAdvice(), mdpPathAdvice=MDPFullPathAdvice(), mdpStateScore=MDPStateScoreDistance(), alpha=200/201, rejectFactor=10, quiet=True, quietInfoStr=True, printEachStep=False, printCompact=True)
	optionsMCTSEngine=OptionsMCTSEngine(horizon=horizon, ignoreNonDecisionStates = True, mctsConstant=math.sqrt(2)/2, mdpActionStrategy=MDPUniformActionStrategy(), mdpActionAdvice=MDPFullActionAdvice(), mdpActionAdviceRoot=mdpActionAdviceRoot, optionsSimulationEngine=optionsSimulationEngine, mdpThresholdActionAdvice = None, mdpThresholdPathAdvice = None, quiet=True, quietInfoStr=True)
	mdp, initState, initPredicates = readFromFile(layout)
	traceEngine = MDPMCTSTraceEngine()
	totalScore = 0
	totalWin = 0
	totalLoss = 0
	totalDraw = 0
	numGames = 0
	for i in range(10):
		results = traceEngine.runMCTSTrace(numTraces = 1, mdpState = initState, mdpPredicates = initPredicates, mdpOperations = mdp, horizonTrace = horizonTrace, numMCTSIters = numMCTSIters, numSims = numSims, optionsMCTSEngine = optionsMCTSEngine, quietTrace = True, quietInfoStr = True, printEachStepTrace = False)
		engineList = [r[0] for r in results]
		for e in engineList:
			# print(e.length(True))
			isWin = "Win" in [p.name for p in e.mdpExecution.mdpPath.mdpPredicatesSequence[-1]]
			isLoss = "Loss" in [p.name for p in e.mdpExecution.mdpPath.mdpPredicatesSequence[-1]]
			totalScore += e.mdpPathReward()
			if isWin:
				totalWin += 1
			elif isLoss:
				totalLoss += 1
			else:
				totalDraw += 1
			numGames += 1
		del results
		# gc.collect()
	avgScore = totalScore/numGames
	avgWin = totalWin/numGames
	avgLoss = totalLoss/numGames
	avgDraw = totalDraw/numGames
	return(avgScore,avgWin,avgLoss,avgDraw)

def getValueFromLayoutNN(layout, model): # use NN and reports average wins
	mdp, initState, initPredicates = readFromFile(layout)
	traceEngine = MDPNNTraceEngine()
	results = traceEngine.runNNTrace(numTraces = 100, mdpState = initState, mdpPredicates = initPredicates, mdpOperations = mdp, horizonTrace = 1000, model = model, threshold = 0.99, quietTrace = True, quietInfoStr = True, printEachStepTrace = False)
	totalScore = 0
	totalWin = 0
	totalLoss = 0
	totalDraw = 0
	numGames = 0
	engineList = [r[0] for r in results]
	for e in engineList:
		isWin = "Win" in [p.name for p in e.mdpExecution.mdpPath.mdpPredicatesSequence[-1]]
		isLoss = "Loss" in [p.name for p in e.mdpExecution.mdpPath.mdpPredicatesSequence[-1]]
		totalScore += e.mdpPathReward()
		if isWin:
			totalWin += 1
		elif isLoss:
			totalLoss += 1
		else:
			totalDraw += 1
		numGames += 1
	del results
	# gc.collect()
	avgScore = totalScore/numGames
	avgWin = totalWin/numGames
	avgLoss = totalLoss/numGames
	avgDraw = totalDraw/numGames
	return(avgScore,avgWin,avgLoss,avgDraw)

def getValueFromLayoutMultiNN(layout, model1, model2, threshold1): # use NN and reports average wins
	mdp, initState, initPredicates = readFromFile(layout)
	traceEngine = MDPMultiNNTraceEngine()
	results = traceEngine.runMultiNNTrace(numTraces = 100, mdpState = initState, mdpPredicates = initPredicates, mdpOperations = mdp, horizonTrace = 1000, model1 = model1, threshold1 = threshold1, model2 = model2, threshold2 = 0.99, quietTrace = True, quietInfoStr = True, printEachStepTrace = False)
	totalScore = 0
	totalWin = 0
	totalLoss = 0
	totalDraw = 0
	numGames = 0
	engineList = [r[0] for r in results]
	for e in engineList:
		isWin = "Win" in [p.name for p in e.mdpExecution.mdpPath.mdpPredicatesSequence[-1]]
		isLoss = "Loss" in [p.name for p in e.mdpExecution.mdpPath.mdpPredicatesSequence[-1]]
		totalScore += e.mdpPathReward()
		if isWin:
			totalWin += 1
		elif isLoss:
			totalLoss += 1
		else:
			totalDraw += 1
		numGames += 1
	del results
	# gc.collect()
	avgScore = totalScore/numGames
	avgWin = totalWin/numGames
	avgLoss = totalLoss/numGames
	avgDraw = totalDraw/numGames
	return(avgScore,avgWin,avgLoss,avgDraw)

def getValueFromLayoutStorm(layout): # use storm and report average wins
	mdp, initState, initPredicates = readFromFile(layout)
	traceEngine = MDPStormTraceEngine()
	totalScore = 0
	totalWin = 0
	totalLoss = 0
	totalDraw = 0
	numGames = 0
	for i in range(100):
		results = traceEngine.runStormTrace(numTraces = 1, mdpState = initState, mdpPredicates = initPredicates, mdpOperations = mdp, horizonTrace = 100, threshold = 0.99, quietTrace = True, quietInfoStr = True, printEachStepTrace = False)
		engineList = [r[0] for r in results]
		for e in engineList:
			isWin = "Win" in [p.name for p in e.mdpExecution.mdpPath.mdpPredicatesSequence[-1]]
			isLoss = "Loss" in [p.name for p in e.mdpExecution.mdpPath.mdpPredicatesSequence[-1]]
			totalScore += e.mdpPathReward()
			if isWin:
				totalWin += 1
			elif isLoss:
				totalLoss += 1
			else:
				totalDraw += 1
			numGames += 1
		del results
		# gc.collect()
	avgScore = totalScore/numGames
	avgWin = totalWin/numGames
	avgLoss = totalLoss/numGames
	avgDraw = totalDraw/numGames
	return(avgScore,avgWin,avgLoss,avgDraw)

def getStormDistDict(layout): # returns a dictionary full of distValues
	mdpOperations, initState, initPredicates = readFromFile(layout)
	d = {}
	walls = mdpOperations.walls
	holes = mdpOperations.holes
	targets = mdpOperations.targets
	height = len(walls)
	width = len(walls[0])

	for i in range(height):
		for j in range(width):
			position = (i,j)
			actionValues = getAllDistValuesFromGrids(walls,holes,targets,position)
			print(position,actionValues)
			d[position] = actionValues
	return d


def getValueFromLayoutDict(layout, horizonTrace): # use storm with distance optimization by creating a dictionary first and report average wins
	mdp, initState, initPredicates = readFromFile(layout)
	traceEngine = MDPDictTraceEngine()
	totalScore = 0
	totalWin = 0
	totalLoss = 0
	totalDraw = 0
	numGames = 0
	d = getStormDistDict(layout)
	for i in range(1):
		results = traceEngine.runDictTrace(numTraces = 100, mdpState = initState, mdpPredicates = initPredicates, mdpOperations = mdp, d = d, horizonTrace = horizonTrace, quietTrace = True, quietInfoStr = True, printEachStepTrace = False)
		engineList = [r[0] for r in results]
		for e in engineList:
			# print(e.length(True))
			isDraw = not e.isTerminal()
			isWin = "Win" in [p.name for p in e.mdpExecution.mdpPath.mdpPredicatesSequence[-1]]
			isLoss = "Loss" in [p.name for p in e.mdpExecution.mdpPath.mdpPredicatesSequence[-1]]
			totalScore += e.mdpPathReward()
			if isDraw:
				totalDraw += 1
			if isWin:
				totalWin += 1
			if isLoss:
				totalLoss += 1
			numGames += 1
		del results
		# gc.collect()
	avgScore = totalScore/numGames
	avgWin = totalWin/numGames
	avgLoss = totalLoss/numGames
	avgDraw = totalDraw/numGames
	return(avgScore,avgWin,avgLoss,avgDraw)

def getValueFromLayoutDictMCTS(layout, horizonTrace , horizon, numMCTSIters, numSims): # use storm with distance optimization by creating a dictionary first and report average wins
	mdp, initState, initPredicates = readFromFile(layout)
	traceEngine = MDPDictTraceEngine()
	totalScore = 0
	totalWin = 0
	totalLoss = 0
	totalDraw = 0
	numGames = 0
	d = getMCTSValueDict(layout, horizon, numMCTSIters, numSims)
	for i in range(1):
		results = traceEngine.runDictTrace(numTraces = 10, mdpState = initState, mdpPredicates = initPredicates, mdpOperations = mdp, d = d, horizonTrace = horizonTrace, quietTrace = True, quietInfoStr = True, printEachStepTrace = False)
		engineList = [r[0] for r in results]
		for e in engineList:
			# print(e.length(True))
			isDraw = not e.isTerminal()
			isWin = "Win" in [p.name for p in e.mdpExecution.mdpPath.mdpPredicatesSequence[-1]]
			isLoss = "Loss" in [p.name for p in e.mdpExecution.mdpPath.mdpPredicatesSequence[-1]]
			totalScore += e.mdpPathReward()
			if isDraw:
				totalDraw += 1
			if isWin:
				totalWin += 1
			if isLoss:
				totalLoss += 1
			numGames += 1
		del results
		# gc.collect()
	avgScore = totalScore/numGames
	avgWin = totalWin/numGames
	avgLoss = totalLoss/numGames
	avgDraw = totalDraw/numGames
	return(avgScore,avgWin,avgLoss,avgDraw)

def getValueFromLayoutStormDist(layout): # use storm with distance optimization and report average wins
	mdp, initState, initPredicates = readFromFile(layout)
	traceEngine = MDPStormDistTraceEngine()
	totalScore = 0
	totalWin = 0
	totalLoss = 0
	totalDraw = 0
	numGames = 0
	for i in range(10):
		results = traceEngine.runStormDistTrace(numTraces = 1, mdpState = initState, mdpPredicates = initPredicates, mdpOperations = mdp, horizonTrace = 100, quietTrace = True, quietInfoStr = True, printEachStepTrace = False)
		engineList = [r[0] for r in results]
		for e in engineList:
			# print(e.length(True))
			isDraw = not e.isTerminal()
			isWin = "Win" in [p.name for p in e.mdpExecution.mdpPath.mdpPredicatesSequence[-1]]
			isLoss = "Loss" in [p.name for p in e.mdpExecution.mdpPath.mdpPredicatesSequence[-1]]
			totalScore += e.mdpPathReward()
			if isDraw:
				totalDraw += 1
			if isWin:
				totalWin += 1
			if isLoss:
				totalLoss += 1
			numGames += 1
		del results
		# gc.collect()
	avgScore = totalScore/numGames
	avgWin = totalWin/numGames
	avgLoss = totalLoss/numGames
	avgDraw = totalDraw/numGames
	return(avgScore,avgWin,avgLoss,avgDraw)

def getValueFromLayoutUniform(layout):# plays uniformly and report average wins
	mdp, initState, initPredicates = readFromFile(layout)
	traceEngine = MDPUniformTraceEngine()
	results = traceEngine.runUniformTrace(numTraces = 100, mdpState = initState, mdpPredicates = initPredicates, mdpOperations = mdp, horizonTrace = 100, quietTrace = True, quietInfoStr = True, printEachStepTrace = False)
	totalScore = 0
	totalWin = 0
	totalLoss = 0
	totalDraw = 0
	numGames = 0
	engineList = [r[0] for r in results]
	for e in engineList:
		isDraw = not e.isTerminal()
		isWin = "Win" in [p.name for p in e.mdpExecution.mdpPath.mdpPredicatesSequence[-1]]
		isLoss = "Loss" in [p.name for p in e.mdpExecution.mdpPath.mdpPredicatesSequence[-1]]
		totalScore += e.mdpPathReward()
		if isDraw:
			totalDraw += 1
		if isWin:
			totalWin += 1
		if isLoss:
			totalLoss += 1
		numGames += 1
	del results
	# gc.collect()
	avgScore = totalScore/numGames
	avgWin = totalWin/numGames
	avgLoss = totalLoss/numGames
	avgDraw = totalDraw/numGames
	return(avgScore,avgWin,avgLoss,avgDraw)

def getNNValue(layout, model): # reports value given by NN
	mdp, initState, initPredicates = readFromFile(layout)
	x = np.expand_dims(mdp.getConfig(initState),axis=0)
	x = np.transpose(x, [0, 2, 3, 1])
	actionValues = keras.backend.get_value(model(x))[0]
	print(actionValues)
	return max(actionValues)

def getMCTSValueDict(layout, horizon, numMCTSIters, numSims): # reports dictionary of values given by mcts
	# create options engline for MCTS
	optionsSimulationEngine=OptionsSimulationEngine(horizon=horizon, ignoreNonDecisionStates = True, mdpActionStrategy=MDPUniformActionStrategy(), mdpActionAdvice=MDPFullActionAdvice(), mdpPathAdvice=MDPFullPathAdvice(), mdpStateScore=MDPStateScoreDistance(), alpha=200/201, rejectFactor=10, quiet=True, quietInfoStr=True, printEachStep=False, printCompact=True)
	optionsMCTSEngine=OptionsMCTSEngine(horizon=horizon, ignoreNonDecisionStates = True, mctsConstant=math.sqrt(2)/2, mdpActionStrategy=MDPUniformActionStrategy(), mdpActionAdvice=MDPFullActionAdvice(), mdpActionAdviceRoot=MDPFullActionAdvice(), optionsSimulationEngine=optionsSimulationEngine, mdpThresholdActionAdvice = None, mdpThresholdPathAdvice = None, quiet=True, quietInfoStr=True)

	# create the MDP ...
	mdp, initState, initPredicates = readFromFile(layout)
	walls = mdp.walls
	height = len(walls)
	width = len(walls[0])

	d = {}
	for i in range(height):
		for j in range(width):
			position = (i,j)
			initState.position = position
			endState = initState.deepCopy()
			mdpPath = MDPPath(initState,[],[initPredicates])
			execEngine = MDPExecutionEngine(mdp,MDPExecution(mdpPath,endState,0,False,1),0)

			if walls[i][j]:
				values = {}
			elif len(mdp.getLegalActions(initState)) == 1:
				values = {}
			else:
				try:
					# run MCTs for 1 step
					mctsEngine = MCTSEngine(execEngine, optionsMCTSEngine)
					mctsEngine.doMCTSIterations(numMCTSIters, numSims)

					# get MCTs values
					values = mctsEngine.getMCTSRootRewardDict()
				except NoMoveException:
					values = {}

			Y = []
			actionList = ["East","West","North","South"]
			for actionName in actionList:
				actionMissing = True
				for k in values.keys():
					if k.action == actionName:
						Y.append((values[k],0))
						actionMissing = False
				if actionMissing:
					Y.append((-1*np.inf,0))
			d[position] = Y
			# print(position,Y)
	return d

def getMCTSValue(layout, horizon, numMCTSIters, numSims): # reports value given by mcts
	# create options engline for MCTS
	optionsSimulationEngine=OptionsSimulationEngine(horizon=horizon, ignoreNonDecisionStates = True, mdpActionStrategy=MDPUniformActionStrategy(), mdpActionAdvice=MDPFullActionAdvice(), mdpPathAdvice=MDPFullPathAdvice(), mdpStateScore=MDPStateScoreDistance(), alpha=200/201, rejectFactor=10, quiet=True, quietInfoStr=True, printEachStep=False, printCompact=True)
	optionsMCTSEngine=OptionsMCTSEngine(horizon=horizon, ignoreNonDecisionStates = True, mctsConstant=math.sqrt(2)/2, mdpActionStrategy=MDPUniformActionStrategy(), mdpActionAdvice=MDPFullActionAdvice(), mdpActionAdviceRoot=MDPFullActionAdvice(), optionsSimulationEngine=optionsSimulationEngine, mdpThresholdActionAdvice = None, mdpThresholdPathAdvice = None, quiet=True, quietInfoStr=True)

	# create the MDP ...
	mdp, initState, initPredicates = readFromFile(layout)
	endState = initState.deepCopy()
	mdpPath = MDPPath(initState,[],[initPredicates])
	execEngine = MDPExecutionEngine(mdp,MDPExecution(mdpPath,endState,0,False,1),0)

	# run MCTs for 1 step
	mctsEngine = MCTSEngine(execEngine, optionsMCTSEngine)
	mctsEngine.doMCTSIterations(numMCTSIters, numSims)

	# get MCTs values
	values = mctsEngine.getMCTSRootRewardDict()
	Y = -np.nan*np.ones(4)
	actionList = ["East","West","North","South"]
	for k in values.keys():
		action = k.action
		try:
			i = actionList.index(action)
			Y[i] = values[k]
		except:
			pass
	return(Y)

def getDirectMCTSValue(mdp, initState, initPredicates, horizon, numMCTSIters, numSims): # reports value given by mcts
	# create options engline for MCTS
	optionsSimulationEngine=OptionsSimulationEngine(horizon=horizon, ignoreNonDecisionStates = True, mdpActionStrategy=MDPUniformActionStrategy(), mdpActionAdvice=MDPFullActionAdvice(), mdpPathAdvice=MDPFullPathAdvice(), mdpStateScore=MDPStateScoreDistance(), alpha=200/201, rejectFactor=10, quiet=True, quietInfoStr=True, printEachStep=False, printCompact=True)
	optionsMCTSEngine=OptionsMCTSEngine(horizon=horizon, ignoreNonDecisionStates = True, mctsConstant=math.sqrt(2)/2, mdpActionStrategy=MDPUniformActionStrategy(), mdpActionAdvice=MDPFullActionAdvice(), mdpActionAdviceRoot=MDPFullActionAdvice(), optionsSimulationEngine=optionsSimulationEngine, mdpThresholdActionAdvice = None, mdpThresholdPathAdvice = None, quiet=True, quietInfoStr=True)

	# create the engine ...
	endState = initState.deepCopy()
	mdpPath = MDPPath(initState,[],[initPredicates])
	execEngine = MDPExecutionEngine(mdp,MDPExecution(mdpPath,endState,0,False,1),0)

	# run MCTs for 1 step
	mctsEngine = MCTSEngine(execEngine, optionsMCTSEngine)
	mctsEngine.doMCTSIterations(numMCTSIters, numSims)

	# get MCTs values
	values = mctsEngine.getMCTSRootRewardDict()
	Y = -np.nan*np.ones(4)
	actionList = ["East","West","North","South"]
	for k in values.keys():
		action = k.action
		try:
			i = actionList.index(action)
			Y[i] = values[k]
		except:
			pass
	del endState, mctsEngine
	return(Y)

def getStormValue(layout): # reprt values given by storm
	values = getAllValuesFromLayout(layout)
	return(np.array(values))

def getStormDistValue(layout): # report distances given by storm if optimal else inf
	values = getAllDistValuesFromLayout(layout)
	maxValue = max([value[0] for value in values])
	distValues = []
	for value in values:
		if value[0] >= maxValue - 0.001:
			distValues.append(value[1])
		else:
			distValues.append(np.nan)
	return(np.array(distValues))

def getAllStormDistValue(layout): # report distances given by storm
	values = getAllDistValuesFromLayout(layout)
	probvalues = [value[0] for value in values]
	distValues = [value[1] for value in values]
	shieldedDistValues = []
	maxValue = max([value[0] for value in values])
	for value in values:
		if value[0] >= maxValue - 0.001:
			shieldedDistValues.append(value[1])
		else:
			shieldedDistValues.append(np.nan)
	return(np.array(probvalues),np.array(distValues),np.array(shieldedDistValues))

def getAllStormDistValueFromGrids(walls,holes,targets,position): # report distances given by storm
	values = getAllDistValuesFromGrids(walls,holes,targets,position)
	probvalues = [value[0] for value in values]
	distValues = [value[1] for value in values]
	shieldedDistValues = []
	maxValue = max([value[0] for value in values])
	for value in values:
		if value[0] >= maxValue - 0.001:
			shieldedDistValues.append(value[1])
		else:
			shieldedDistValues.append(np.nan)
	return(np.array(probvalues),np.array(distValues),np.array(shieldedDistValues))

def getBoundedStormValue(layout,distance=None): # report values given by storm
	if distance == None:
		value = getValueFromLayout(layout)
	else:
		value = getValueFromLayout(layout,formula_str = f"Pmax=? [F<={distance} win]")
	return(value)

def getAllValuesInDirStorm(dirName,prefix,id):
	files = glob.glob(dirName+os.sep+prefix+"_**.lay")
	f = open(dirName+os.sep+prefix+f"_storm_new_result_1000.csv","w+")
	print("layout, stormValue, avgScore , avgWin , avgLoss , avgDraw ",file=f)#+" Storm, Uniform"
	for file in files:
		print(file)
		try:
			v1 = getBoundedStormValue(file)
			avgScore,avgWin,avgLoss,avgDraw = getValueFromLayoutStormDist(file)
			s = file.split('/')[-1] + ',' + str(v1) + ',' + str(avgScore) + ',' + str(avgWin) + ',' + str(avgLoss)  + ',' + str(avgDraw)
			print(s)
			print(s,file=f)
		except Exception as e:
			print(e)
	f.close()

def getAllValuesInDirDict(dirName,prefix,horizonTrace):
	files = glob.glob(dirName+os.sep+prefix+"_**.lay")
	f = open(dirName+os.sep+prefix+f"_stormNew_result_{horizonTrace}.csv","w+")
	print("layout, stormValue, avgScore , avgWin , avgLoss , avgDraw ",file=f)#+" Storm, Uniform"
	for file in files:
		print(file)
		try:
			v1 = getBoundedStormValue(file)
			avgScore,avgWin,avgLoss,avgDraw = getValueFromLayoutDict(file,horizonTrace)
			s = file.split('/')[-1] + ',' + str(v1) + ',' + str(avgScore) + ',' + str(avgWin) + ',' + str(avgLoss)  + ',' + str(avgDraw)
			print(s)
			print(s,file=f)
		except Exception as e:
			print(e)
	f.close()

def getAllValuesInDirDictMCTS(dirName,prefix,id,horizonTrace, mctsParams):
	files = glob.glob(dirName+os.sep+prefix+"_**.lay")
	for mctsParam in mctsParams:
		horizon  = mctsParam[0]
		numMCTSIters  = mctsParam[1]
		numSims  = mctsParam[2]
		f = open(dirName+os.sep+f"{prefix}_{id}_MCTS_result_{horizon}_{numMCTSIters}_{numSims}_{horizonTrace}.csv","w+")
		print("layout, stormValue, avgScore , avgWin , avgLoss , avgDraw ",file=f)#+" Storm, Uniform"
		for file in files:
			print(file)
			try:
				v1 = getBoundedStormValue(file)
				avgScore,avgWin,avgLoss,avgDraw = getValueFromLayoutDictMCTS(file,horizonTrace, horizon, numMCTSIters, numSims)
				s = file.split('/')[-1] + ',' + str(v1) + ',' + str(avgScore) + ',' + str(avgWin) + ',' + str(avgLoss)  + ',' + str(avgDraw)
				print(s)
				print(s,file=f)
			except Exception as e:
				print(e)
		f.close()

def getAllValuesInDirUniform(dirName,prefix):
	files = glob.glob(dirName+os.sep+prefix+"_**.lay")
	f = open(dirName+os.sep+prefix+"_uniform_result.csv","w+")
	print("layout, stormValue, avgScore , avgWin , avgLoss , avgDraw ",file=f)#+" Storm, Uniform"
	for file in files:
		print(file)
		try:
			v1 = getBoundedStormValue(file)
			avgScore,avgWin,avgLoss,avgDraw = getValueFromLayoutUniform(file)
			s = file.split('/')[-1] + ',' + str(v1) + ',' + str(avgScore) + ',' + str(avgWin) + ',' + str(avgLoss)  + ',' + str(avgDraw)
			print(s)
			print(s,file=f)
		except Exception as e:
			print(e)
	f.close()

def getAllValuesInDirMCTS(dirName,mctsParams,prefix,id,horizonTrace):
	files = glob.glob(dirName+os.sep+prefix+"_**.lay")
	for mctsParam in mctsParams:
		horizon  = mctsParam[0]
		numMCTSIters  = mctsParam[1]
		numSims  = mctsParam[2]
		f = open(dirName+os.sep+f"{prefix}_{id}_MCTS_result_{horizon}_{numMCTSIters}_{numSims}_{horizonTrace}.csv","w+")
		print("layout, stormValue, avgScore , avgWin , avgLoss , avgDraw ",file=f)#+" Storm, Uniform"
		for file in files:
			print(file, "mcts")
			try:
				v1 = getBoundedStormValue(file)
				#v2 = getBoundedStormValue(file, 100)
				avgScore,avgWin,avgLoss,avgDraw = getValueFromLayoutMCTS(file, horizon, numMCTSIters, numSims,horizonTrace)
				s = file.split('/')[-1] + ',' + str(v1) + ',' + str(avgScore) + ',' + str(avgWin) + ',' + str(avgLoss)  + ',' + str(avgDraw)
				print(s)
				print(s,file=f)
			except Exception as e:
				print(e)
		f.close()

def getAllValuesInDirNN(dirName,NNList,prefix):
	files = glob.glob(dirName+os.sep+prefix+"_**.lay")
	for NN in NNList:
		print(NN)
		modelFile = MODEL_DIR+os.sep+f"actiondata_{NN}.h5"
		model = keras.models.load_model(modelFile,compile = False)
		f = open(dirName+os.sep+f"{prefix}_NN_result_{NN}.csv","w+")
		print("layout, stormValue, avgScore , avgWin , avgLoss , avgDraw ",file=f)#+" Storm, Uniform"
		for file in files:
			print(file)
			try:
				v1 = getBoundedStormValue(file)
				avgScore,avgWin,avgLoss,avgDraw = getValueFromLayoutNN(file, model)
				s = file.split('/')[-1] + ',' + str(v1) + ',' + str(avgScore) + ',' + str(avgWin) + ',' + str(avgLoss)  + ',' + str(avgDraw)
				print(s)
				print(s,file=f)
			except Exception as e:
				print(e)
		f.close()

def getAllValuesInDirMultiNN(dirName,NNParams,prefix):
	files = glob.glob(dirName+os.sep+prefix+"_**.lay")
	for param in NNParams:
		print(param)
		NN1 = param[0]
		NN2 = param[1]
		threshold1 = param[2]
		modelFile1 = MODEL_DIR+os.sep+f"actiondata_{NN1}.h5"
		model1 = keras.models.load_model(modelFile1,compile = False)
		modelFile2 = MODEL_DIR+os.sep+f"actiondata_{NN2}.h5"
		model2 = keras.models.load_model(modelFile2,compile = False)
		f = open(dirName+os.sep+f"{prefix}_NN_result_{NN1}_t{int(threshold1*1000)}_{NN2}.csv","w+")
		print("layout, stormValue, avgScore , avgWin , avgLoss , avgDraw ",file=f)#+" Storm, Uniform"
		for file in files:
			print(file)
			try:
				v1 = getBoundedStormValue(file)
				avgScore,avgWin,avgLoss,avgDraw = getValueFromLayoutMultiNN(file, model1, model2, threshold1)
				s = file.split('/')[-1] + ',' + str(v1) + ',' + str(avgScore) + ',' + str(avgWin) + ',' + str(avgLoss)  + ',' + str(avgDraw)
				print(s)
				print(s,file=f)
			except Exception as e:
				print(e)
		f.close()

def getAllStormDistValues(dirName):
	files = glob.glob(dirName+os.sep+"**_**.lay")
	f = open(dirName+os.sep+f"distValues.csv","w+")
	for file in files:
		print(file)
		try:
			v1 = getBoundedStormValue(file)
			v2 = np.nanmin(getStormDistValue(file))
			s = file.split('/')[-1] + ',' + str(v1) + ',' + str(v2)
			print(s)
			print(s,file=f)
		except Exception as e:
			print(e)
	f.close()

# def getAllValues(layout,mctsParams,modelFiles):
# 	s = layout + ','
# 	for mctsParam in mctsParams:
# 		s += str(getValueFromLayoutMCTS(layout,mctsParam[0],mctsParam[1],mctsParam[2])) + ','
# 	for modelFile in modelFiles:
# 		model = keras.models.load_model(modelFile,compile = False)
# 		s += str(getValueFromLayoutNN(layout, model)) + ','
# 	# s += str(getValueFromLayout(layout)) + ','
# 	# s += str(getValueFromLayoutUniform(layout)) + ','
# 	return s
#
# def getAllValuesInDir(dirName,mctsParams,modelFiles,prefix):
# 	files = glob.glob(dirName+os.sep+prefix+"_**.lay")
# 	f = open(dirName+os.sep+prefix+"_result.csv","w+")
# 	print("layout,"+" MCTS,"*len(mctsParams)+" NN,"*len(modelFiles),file=f)#+" Storm, Uniform"
# 	for file in files:
# 		print(file)
# 		try:
# 			s = getAllValues(file,mctsParams,modelFiles)
# 		except NoMoveException:
# 			pass
# 		print(s,file=f)
# 	f.close()
#
# def getAllValuesInDirStormRoot(dirName,prefix):
# 	files = glob.glob(dirName+os.sep+prefix+"_**.lay")
# 	with open(dirName+os.sep+prefix+"_storm_root_result.csv","w+") as f:
# 		print("layout, storm sim, storm mcts",file=f)
# 		for file in files:
# 			print(file)
# 			try:
# 				v1 = getValueFromLayoutStorm(file)
# 				v2 = getValueFromLayoutMCTS(file, 1, 40, 10, stormAtRoot = True)
# 				s = file.split('/')[-1] + ',' + str(v1) + ',' + str(v2)
# 			except Exception as e:
# 				print(e)
# 			print(s,file=f)

# def getAllValuesInDirStormRoot(dirName,prefix):
# 	files = glob.glob(dirName+os.sep+prefix+"_**.lay")
# 	f = open(dirName+os.sep+prefix+"_storm_root_result.csv","w+")
# 	print("layout, onlyStorm, stormDist",file=f)
# 	for file in files:
# 		print(file)
# 		try:
# 			s = file.split('/')[-1] + ',' + str(getValueFromLayoutStorm(file)) + ',' + str(getValueFromLayoutStorm(file))
# 		except NoMoveException:
# 			pass
# 		print(s,file=f)
# 	f.close()



if __name__ == "__main__":
	# prefix = sys.argv[1]
	# dirName = LAYOUTS_DIR
	# layouts = glob.glob(dirName+os.sep+prefix+"_**.lay")
	# for layout in layouts:
	# 	print(layout)
	# # 	getStormDistDict(layout)
	# 	getMCTSValueDict(layout, 30, 40, 10)



	prefix = sys.argv[1]
	id = int(sys.argv[2])
	horizonTrace = 1000
	dirName = LAYOUTS_DIR
	util.mkdir(LAYOUTS_DIR)
	# getAllValuesInDirDict(dirName,prefix,horizonTrace)
	# # if id == 0:
	# getAllValuesInDirStorm(dirName,prefix,id)
	# # elif id == 1:
	# # 	getAllValuesInDirUniform(dirName,prefix)
	# # elif id == 2:
	# mctsParams = [(30,40,10),(1,40,30)]
	# getAllValuesInDirMCTS(dirName,mctsParams,prefix,id,horizonTrace)
	# getAllValuesInDirDictMCTS(dirName,prefix,id,horizonTrace, mctsParams)
	# getAllStormDistValues(dirName)
	# elif id == 3:
	NNList = ["random_MCTS_h30_i40_s10_7"]#"1_40_30","30_40_10","StormShielded""random_StormShielded_7"
	getAllValuesInDirNN(dirName,NNList,prefix)
	# NNParams = [("StormProb","StormDist",0.99),("StormProb","StormDist",0.95),("StormProb","StormDist",0.9)]
	# getAllValuesInDirMultiNN(dirName,NNParams,prefix)
	# else:
	# 	raise Exception(f"Wrong id: {id}")
