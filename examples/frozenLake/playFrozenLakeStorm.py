# playFrozenLake.py

import sys, argparse, math, os

CURRENTPWD = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(CURRENTPWD, '../src'))

from adviceMCTS.stormMdpClasses import *
# from frozenLake import fullGridStrPosition
# from frozenLakeMdpClasses import gridsFromFile, normalizeDistance, normalizeFloat, MDPNNTraceEngine
# from stormMdpClasses import runGamesWithMCTS
from frozenLakeStorm import *
import json

# Grid = List[List[bool]]
# GridDistance = List[List[int]]

def gridDistance(walls, holes, targetList):
	X=len(walls)
	Y=len(walls[0])
	infty=X*Y+1
	r = [[ infty for j in range(Y)] for i in range(X)]
	queue = []
	maxd=infty
	for x,y in targetList:
		r[x][y]=0
		queue.append((x,y,0))
		maxd=0
	while len(queue)>0:
		x,y,d = queue.pop(0)
		for xx,yy in [(x-1,y),(x,y-1),(x+1,y),(x,y+1)]:
			if xx>=0 and xx<X and yy>=0 and yy<Y and (not walls[xx][yy]) and (not holes[xx][yy]) and r[xx][yy]>d+1:
				r[xx][yy]=d+1
				queue.append((xx,yy,d+1))
				if d+1>maxd:
					maxd=d+1
	return (r,maxd)

def runResults(engineList: List[MDPExecutionEngine[MDPPredicate, MDPState, MDPAction, MDPStochasticAction]], quiet: bool = True, prettyConsole: bool = False) -> Any:
	print("[== running replay engines")
	for mdpExecutionEngine in engineList:
		cursesScr = None
		cursesDelay = 0.0
		if prettyConsole:
			cursesScr = curses.initscr()
			curses.noecho()
			curses.cbreak()
			cursesDelay = min(0.5,max(0.05,60.0/mdpExecutionEngine.length(ignoreNonDecisionStates=False)))

		optionsReplayEngine=OptionsReplayEngine(quiet=quiet, printEachStep=False, printCompact=False, cursesScr=cursesScr, cursesDelay=cursesDelay)
		mdpReplayEngine = MDPReplayEngine(mdpExecutionEngine.mdpOperations,mdpExecutionEngine.mdpPath(),options=optionsReplayEngine)
		try:
			while not mdpReplayEngine.isTerminal():
				mdpReplayEngine.advanceReplay()
			if (mdpReplayEngine.mdpExecutionEngine.isTerminal() != mdpExecutionEngine.isTerminal()) or (mdpReplayEngine.mdpExecutionEngine.mdpPathReward() != mdpExecutionEngine.mdpPathReward()) or (mdpReplayEngine.mdpExecutionEngine.mdpEndState() != mdpExecutionEngine.mdpEndState()):
				raise Exception("bad replay")
			if prettyConsole:
				curses.endwin()
		except BaseException as error:
			if prettyConsole:
				curses.endwin()
			raise Exception(str(error))


		# mdpReplayEngine.resetReplay()
	print("==] done")


# def readCommand(argv):
# 	usageStr =  """
# 		USAGE:      python3 playFrozenLakeStorm.py <options>
# 		EXAMPLE:    python3 playFrozenLakeStorm.py -l layouts/frozenLake/test.lay -n 1 -s 100 -iter 50 -sim 10 -H 20 -psT -qm -qs -qi
# 		"""
#
# 	parser = argparse.ArgumentParser(usageStr)
# 	parser.add_argument('-v', '--verbose', dest='verbose', action='count', default=0)
# 	parser.add_argument('-l', '-lay', '--layout', dest='layout', type=str, help='the LAYOUT', metavar='LAYOUT')
# 	parser.add_argument('-n', '-nTr', '--numTraces', dest='numTraces', type=int, help='the number of TRACES to play (Default: 1)', metavar='TRACES', default=1)
# 	parser.add_argument('-s', '-step', '--steps', dest='horizonTrace', type=int, help='the number of STEPS to play in each trace(Default: 50)', metavar='STEPS', default=50)
# 	parser.add_argument('-iter', '--numMCTSIters', dest='numMCTSIters', type=int, help='the number of ITERATIONS in MCTS(Default: 50)', metavar='ITERATIONS', default=50)
# 	parser.add_argument('-sim', '--numMCTSSims', dest='numSims', type=int, help='the number of SIMULATIONS in each MCTS iterations(Default: 10)', metavar='SIMULATIONS', default=10)
# 	parser.add_argument('-H', '--horizon', dest='horizon', type=int, help='HORIZON for MCTS (Default: 20)', metavar='HORIZON', default=20)
# 	parser.add_argument('--ignoreNonDecisionStates', dest='ignoreNonDecisionStates', action='store_true', default=False, help='If True, states with a single action are ignored for horizon purposes')
# 	parser.add_argument('--MCTSActionAdvice', dest='mctsActionAdvice', type=str, help='MCTS Action ADVICE for selection(Default: Full)', metavar='ADVICE')
# 	parser.add_argument('--MCTSActionAdviceSim', dest='mctsActionAdviceSim', type=str, help='MCTS Action ADVICE for simulation(Default: Full)', metavar='ADVICE')
# 	parser.add_argument('--MCTSPathAdviceSim', dest='mctsPathAdviceSim', type=str, help='MCTS Path ADVICE for simulation(Default: Full)', metavar='ADVICE')
# 	parser.add_argument('-C', '--MCTSConstant', dest='mctsConstant', type=float, help='MCTS CONSTANT (write in float; Default: math.sqrt(2)/2)', metavar='CONSTANT')
# 	parser.add_argument('--MCTSStateScore', dest='mctsStateScore', type=str, help='MCTS State Score for simulation (Default: Zero)', metavar='SCORE')
# 	parser.add_argument('--StateScoreModel', dest='modelStateScore', type=str, help='Model file for State Score', metavar='MODEL')
# 	parser.add_argument('-a', '--stateScoreWeight', dest='alpha', type=float, help='MCTS State Score weight (write in float; Default: 0.5', metavar='CONSTANT', default=0.5)
# 	parser.add_argument('--discount', dest='discount', type=float, help='Discount in MDP (Default: 1.0)', metavar='CONSTANT', default=1.0)
# 	parser.add_argument('-qm', '--quietMCTS', dest='quietMCTS', action='store_true', default=None, help='Does not print console outputs with doing MCTS')
# 	parser.add_argument('-qt', '--quietTrace', dest='quietTrace', action='store_true', default=None, help='Does not print the trace')
# 	parser.add_argument('-qs', '--quietSim', dest='quietSim', action='store_true', default=None, help='Does not print the sims')
# 	parser.add_argument('-ps', '--printEachStep', dest='printEachStep', action='store_true', default=None, help='Print each step of the sims')
# 	parser.add_argument('-psT', '--printEachStepTrace', dest='printEachStepTrace', action='store_true', default=None,    help='Print each step of the trace')
# 	parser.add_argument('-qi', '--quietInfostr', dest='quietInfoStr', action='store_true', default=None, help='Does not print additional information related to actions')
# 	parser.add_argument('-r', '--replay', dest='replay', action='store_true', default=False, help='Replays the game')
#
# 	options = parser.parse_args(argv)
# 	args = {}
# 	if options.layout is None:
# 		args['layout'] = 'layouts'+os.sep+'frozenLake'+os.sep+'test.lay'
# 	else:
# 		args['layout'] = options.layout
# 	args['numTraces'] = options.numTraces
# 	args['horizonTrace'] = options.horizonTrace
# 	args['numMCTSIters'] = options.numMCTSIters
# 	args['numSims'] = options.numSims
#
# 	args['replay'] = options.replay
#
# 	if options.mctsConstant is None:
# 		mctsConstant = math.sqrt(2)/2
# 	else:
# 		mctsConstant = options.mctsConstant
#
# 	# mdpStateScore=MDPStateScore()
# 	if options.mctsStateScore is None:
# 		mdpStateScore=MDPStateScore()
# 	elif options.mctsStateScore == "Zero":
# 		mdpStateScore=MDPStateScoreZero()
# 	elif options.mctsStateScore == "Distance":
# 		scoreFunction = frozenLake(args['layout']).getScore
# 		mdpStateScore=MDPStateScoreDistance(scoreFunction)
# 	elif options.mctsStateScore == "NN":
# 		from tensorflow import keras
# 		modelFile = options.modelStateScore
# 		model = keras.models.load_model(modelFile)
# 		mdpStateScore=MDPStateScoreNN(model)
# 	else:
# 		raise Exception ("Unknown state score")
#
# 	alpha=options.alpha
#
# 	if options.mctsActionAdvice is None:
# 		mdpActionAdvice = MDPFullActionAdvice()
# 	elif options.mctsActionAdvice == "Full":
# 		mdpActionAdvice = MDPFullActionAdvice()
# 	elif options.mctsActionAdvice == "EXNonLoss":
# 		mdpActionAdvice = MDPEXNonLossActionAdvice()
# 	elif options.mctsActionAdvice == "AXNonLoss":
# 		mdpActionAdvice = MDPAXNonLossActionAdvice()
# 	else:
# 		raise Exception ("Unknown action advice for selection")
#
# 	if options.mctsActionAdviceSim is None:
# 		mdpActionAdviceSim = MDPFullActionAdvice()
# 	elif options.mctsActionAdviceSim == "Full":
# 		mdpActionAdviceSim = MDPFullActionAdvice()
# 	elif options.mctsActionAdviceSim == "EXNonLoss":
# 		mdpActionAdviceSim = MDPEXNonLossActionAdvice()
# 	elif options.mctsActionAdviceSim == "AXNonLoss":
# 		mdpActionAdviceSim = MDPAXNonLossActionAdvice()
# 	else:
# 		raise Exception ("Unknown action advice for simulation")
#
# 	if options.mctsPathAdviceSim is None:
# 		mdpPathAdviceSim = MDPFullPathAdvice()
# 	elif options.mctsPathAdviceSim == "Full":
# 		mdpPathAdviceSim = MDPFullPathAdvice()
# 	# elif options.mctsPathAdviceSim == "NonLoss":
# 		# mdpPathAdviceSim = MDPNonLossPathAdvice()
# 	else:
# 		raise Exception ("Unknown path advice for simulation")
#
# 	if options.verbose == 0:
# 		quietMCTS=True
# 		quietTrace=True
# 		quietSim=True
# 		printEachStep=False
# 		printEachStepTrace=False
# 	elif options.verbose == 1:
# 		quietMCTS=True
# 		quietTrace=False
# 		quietSim=True
# 		printEachStep=False
# 		printEachStepTrace=False
# 	elif options.verbose == 2:
# 		quietMCTS=True
# 		quietTrace=False
# 		quietSim=True
# 		printEachStep=False
# 		printEachStepTrace=True
# 	elif options.verbose == 3:
# 		quietMCTS=False
# 		quietTrace=False
# 		quietSim=True
# 		printEachStep=False
# 		printEachStepTrace=True
# 	elif options.verbose == 4:
# 		quietMCTS=False
# 		quietTrace=False
# 		quietSim=False
# 		printEachStep=False
# 		printEachStepTrace=True
# 	else:
# 		quietMCTS=False
# 		quietTrace=False
# 		quietSim=False
# 		printEachStep=True
# 		printEachStepTrace=True
# 	if not options.quietMCTS is None:
# 		quietMCTS=options.quietMCTS
# 	if not options.quietTrace is None:
# 		quietTrace=options.quietTrace
# 	if not options.quietSim is None:
# 		quietSim=options.quietSim
# 	if not options.printEachStep is None:
# 		printEachStep=options.printEachStep
# 	if not options.printEachStepTrace is None:
# 		printEachStepTrace=options.printEachStepTrace
#
# 	optionsSimulationEngine=OptionsSimulationEngine(horizon=options.horizon, ignoreNonDecisionStates = options.ignoreNonDecisionStates, mdpActionStrategy=MDPUniformActionStrategy(), mdpActionAdvice=mdpActionAdviceSim, mdpPathAdvice=mdpPathAdviceSim, mdpStateScore=mdpStateScore, alpha=alpha, rejectFactor=10, quiet=quietSim, quietInfoStr=options.quietInfoStr, printEachStep=printEachStep, printCompact=True)
#
# 	optionsMCTSEngine=OptionsMCTSEngine(horizon=options.horizon, ignoreNonDecisionStates = options.ignoreNonDecisionStates, mctsConstant=mctsConstant, mdpActionStrategy=MDPUniformActionStrategy(), mdpActionAdvice=mdpActionAdvice, mdpActionAdviceRoot=mdpActionAdvice, optionsSimulationEngine=optionsSimulationEngine, mdpThresholdActionAdvice = None, mdpThresholdPathAdvice = None, quiet=quietMCTS, quietInfoStr=options.quietInfoStr)
#
# 	args['discount'] = options.discount
#
# 	args['optionsMCTSEngine'] = optionsMCTSEngine
# 	args['quietTrace'] = quietTrace
# 	args['quietInfoStr'] = options.quietInfoStr
# 	args['printEachStepTrace'] = printEachStepTrace
#
#
# 	return (args)

def readCommand(argv):
	usageStr =  """
		USAGE:      python3 playFrozenLake.py <options>
		EXAMPLE:    python3 playFrozenLake.py -l layouts/frozenLake/test.lay --ignoreNonDecisionStates -n 1 -s 100 -iter 50 -sim 10 -H 20 -psT -qm -qs -qi
		"""

	parser = argparse.ArgumentParser(usageStr)

	mode = parser.add_mutually_exclusive_group()
	mode.add_argument('--mcts', dest='useMCTS', action='store_true', default=False, help='Uses mcts')
	mode.add_argument('--dt', dest='useDT', action='store_true', default=False, help='Uses a decision tree')
	mode.add_argument('--nn', dest='useNN', action='store_true', default=False, help='Uses a neural network')

	parser.add_argument('-v', '--verbose', dest='verbose', action='count', default=0)
	parser.add_argument('-l', '-lay', '--layout', dest='layout', type=str, help='the LAYOUT', metavar='LAYOUT')
	parser.add_argument('-n', '-nTr', '--numTraces', dest='numTraces', type=int, help='the number of TRACES to play (Default: 1)', metavar='TRACES', default=1)
	parser.add_argument('-s', '-step', '--steps', dest='horizonTrace', type=int, help='the number of STEPS to play in each trace(Default: 50)', metavar='STEPS', default=50)
	parser.add_argument('--discount', dest='discount', type=float, help='Discount in MDP (Default: 1.0)', metavar='CONSTANT', default=1.0)
	parser.add_argument('--decisionTree', dest='decisionTree', type=str, help='decision TREE file for actions', metavar='TREE')
	parser.add_argument('--nnModel', dest='nnModel', type=str, help='neural network MODEL file for actions', metavar='MODEL')
	parser.add_argument('-iter', '--numMCTSIters', dest='numMCTSIters', type=int, help='the number of ITERATIONS in MCTS(Default: 50)', metavar='ITERATIONS', default=50)
	parser.add_argument('-sim', '--numMCTSSims', dest='numSims', type=int, help='the number of SIMULATIONS in each MCTS iterations(Default: 10)', metavar='SIMULATIONS', default=10)
	parser.add_argument('-H', '--horizon', dest='horizon', type=int, help='HORIZON for MCTS (Default: 20)', metavar='HORIZON', default=20)
	parser.add_argument('--ignoreNonDecisionStates', dest='ignoreNonDecisionStates', action='store_true', default=False, help='If set, states with a single action are ignored for horizon purposes')
	parser.add_argument('--MCTSActionAdvice', dest='mctsActionAdvice', type=str, help='MCTS Action ADVICE for selection(Default: Full)', metavar='ADVICE')
	parser.add_argument('--MCTSActionAdviceSim', dest='mctsActionAdviceSim', type=str, help='MCTS Action ADVICE for simulation(Default: Full)', metavar='ADVICE')
	parser.add_argument('--MCTSPathAdviceSim', dest='mctsPathAdviceSim', type=str, help='MCTS Path ADVICE for simulation(Default: Full)', metavar='ADVICE')
	parser.add_argument('-C', '--MCTSConstant', dest='mctsConstant', type=float, help='MCTS CONSTANT (write in float; Default: math.sqrt(2)/2)', metavar='CONSTANT')
	parser.add_argument('--MCTSStateScore', dest='mctsStateScore', type=str, help='MCTS State Score for simulation (Default: Zero)', metavar='SCORE')
	parser.add_argument('--StateScoreModel', dest='modelStateScore', type=str, help='Model file for State Score', metavar='MODEL')
	parser.add_argument('-a', '--stateScoreWeight', dest='alpha', type=float, help='MCTS State Score weight (write in float; Default: 0.5', metavar='CONSTANT', default=0.5)
	parser.add_argument('-qm', '--quietMCTS', dest='quietMCTS', action='store_true', default=None, help='Does not print console outputs with doing MCTS')
	parser.add_argument('-qt', '--quietTrace', dest='quietTrace', action='store_true', default=None, help='Does not print the trace')
	parser.add_argument('-qs', '--quietSim', dest='quietSim', action='store_true', default=None, help='Does not print the sims')
	parser.add_argument('-ps', '--printEachStep', dest='printEachStep', action='store_true', default=None, help='Print each step of the sims')
	parser.add_argument('-psT', '--printEachStepTrace', dest='printEachStepTrace', action='store_true', default=None,	help='Print each step of the trace')
	parser.add_argument('-qi', '--quietInfostr', dest='quietInfoStr', action='store_true', default=None, help='Does not print additional information related to actions')
	parser.add_argument('-r', '--replay', dest='replay', action='store_true', default=False, help='Replays the game')
	options = parser.parse_args(argv)
	args = {}
	args['layout'] = options.layout
	args['numTraces'] = options.numTraces
	args['horizonTrace'] = options.horizonTrace
	args['discount'] = options.discount

	args['useMCTS'] = options.useMCTS
	args['useDT'] = options.useDT
	args['useNN'] = options.useNN

	if options.verbose == 0:
		quietMCTS=True
		quietTrace=True
		quietSim=True
		printEachStep=False
		printEachStepTrace=False
	elif options.verbose == 1:
		quietMCTS=True
		quietTrace=False
		quietSim=True
		printEachStep=False
		printEachStepTrace=False
	elif options.verbose == 2:
		quietMCTS=True
		quietTrace=False
		quietSim=True
		printEachStep=False
		printEachStepTrace=True
	elif options.verbose == 3:
		quietMCTS=False
		quietTrace=False
		quietSim=True
		printEachStep=False
		printEachStepTrace=True
	elif options.verbose == 4:
		quietMCTS=False
		quietTrace=False
		quietSim=False
		printEachStep=False
		printEachStepTrace=True
	else:
		quietMCTS=False
		quietTrace=False
		quietSim=False
		printEachStep=True
		printEachStepTrace=True
	if not options.quietMCTS is None:
		quietMCTS=options.quietMCTS
	if not options.quietTrace is None:
		quietTrace=options.quietTrace
	if not options.quietSim is None:
		quietSim=options.quietSim
	if not options.printEachStep is None:
		printEachStep=options.printEachStep
	if not options.printEachStepTrace is None:
		printEachStepTrace=options.printEachStepTrace

	args['quietTrace'] = quietTrace
	args['quietInfoStr'] = options.quietInfoStr
	args['printEachStepTrace'] = printEachStepTrace

	if options.useMCTS:
		args['horizonTrace'] = options.horizonTrace
		args['numMCTSIters'] = options.numMCTSIters
		args['numSims'] = options.numSims

		if options.mctsConstant is None:
			mctsConstant = math.sqrt(2)/2
		else:
			mctsConstant = options.mctsConstant

		# mdpStateScore=MDPStateScore()
		if options.mctsStateScore is None:
			mdpStateScore=MDPStateScore()
		elif options.mctsStateScore == "Zero":
			mdpStateScore=MDPStateScoreZero()
		elif options.mctsStateScore == "Distance":
			mdpStateScore=MDPStateScoreSimple()
		elif options.mctsStateScore == "NN":
			from tensorflow import keras
			modelFile = options.modelStateScore
			layout = args['layout']
			model = keras.models.load_model(modelFile)
			mdpStateScore=MDPStateScoreNN(model,layout)
		else:
			raise Exception ("Unknown state score")

		alpha=options.alpha

		if options.mctsActionAdvice is None:
			mdpActionAdvice = MDPFullActionAdvice()
		elif options.mctsActionAdvice == "Full":
			mdpActionAdvice = MDPFullActionAdvice()
		elif options.mctsActionAdvice == "EXNonLoss":
			mdpActionAdvice = MDPEXNonLossActionAdvice()
		elif options.mctsActionAdvice == "AXNonLoss":
			mdpActionAdvice = MDPAXNonLossActionAdvice()
		else:
			raise Exception ("Unknown action advice for selection")

		if options.mctsActionAdviceSim is None:
			mdpActionAdviceSim = MDPFullActionAdvice()
		elif options.mctsActionAdviceSim == "Full":
			mdpActionAdviceSim = MDPFullActionAdvice()
		elif options.mctsActionAdviceSim == "EXNonLoss":
			mdpActionAdviceSim = MDPEXNonLossActionAdvice()
		elif options.mctsActionAdviceSim == "AXNonLoss":
			mdpActionAdviceSim = MDPAXNonLossActionAdvice()
		else:
			raise Exception ("Unknown action advice for simulation")

		## TODO: mdpactionadviceroot
		mdpActionAdviceRoot = MDPFullActionAdvice()

		if options.mctsPathAdviceSim is None:
			mdpPathAdviceSim = MDPFullPathAdvice()
		elif options.mctsPathAdviceSim == "Full":
			mdpPathAdviceSim = MDPFullPathAdvice()
		elif options.mctsPathAdviceSim == "NonLoss":
			mdpPathAdviceSim = MDPNonLossPathAdvice()
		else:
			raise Exception ("Unknown path advice for simulation")

		optionsSimulationEngine=OptionsSimulationEngine(horizon=options.horizon, ignoreNonDecisionStates = options.ignoreNonDecisionStates, mdpActionStrategy=MDPUniformActionStrategy(), mdpActionAdvice=mdpActionAdviceSim, mdpPathAdvice=mdpPathAdviceSim, mdpStateScore=mdpStateScore, alpha=alpha, rejectFactor=10, quiet=quietSim, quietInfoStr=options.quietInfoStr, printEachStep=printEachStep, printCompact=True)
		optionsMCTSEngine=OptionsMCTSEngine(horizon=options.horizon, ignoreNonDecisionStates = options.ignoreNonDecisionStates, mctsConstant=mctsConstant, mdpActionStrategy=MDPUniformActionStrategy(), mdpActionAdvice=mdpActionAdvice, mdpActionAdviceRoot=mdpActionAdviceRoot, optionsSimulationEngine=optionsSimulationEngine, mdpThresholdActionAdvice = None, mdpThresholdPathAdvice = None, quiet=quietMCTS, quietInfoStr=options.quietInfoStr)

		args['optionsMCTSEngine'] = optionsMCTSEngine

	elif options.useDT:
		from sklearn import tree
		dtFile = options.decisionTree
		f = open(dtFile, 'rb')
		decisionTree = pickle.load(f)
		args['tree'] = decisionTree
	elif options.useNN:
		from tensorflow import keras
		modelFile = options.nnModel
		model = keras.models.load_model(modelFile)
		args['model'] = model
		args['threshold'] = options.nnThreshold
	elif options.useStorm:
		args['threshold'] = options.stormThreshold
	elif options.useStormDist:
		pass
	else:
		raise SystemExit(": error: use one of these arguments to select the mode: [--mcts | --dt | --nn]")

	args['replay'] = options.replay

	return (args)

# class MDPFullPathAdvice(MDPPathAdviceInterface[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction]):
#     def isValidPath(self, mdpExecutionEngine: MDPExecutionEngine[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction]) -> bool:
#         return True

class frozenLake:

	def __init__(self,layout):
		walls, holes, targets, position = gridsFromFile(layout)
		self.walls = walls
		self.holes = holes
		self.targets=targets
		self.initPos=position
		self.targetList=[]
		self.holeList=[]
		for i in range(len(self.targets)):
			for j in range(len(self.targets[i])):
				if targets[i][j]:
					self.targetList.append((i,j))
		for i in range(len(self.holes)):
			for j in range(len(self.holes[i])):
				if self.holes[i][j]:
					self.holeList.append((i,j))
		self.holeDistance,self.maxHoleDistance = gridDistance(walls,targets,self.holeList)
		self.targetDistance,self.maxTargetDistance = gridDistance(walls,holes,self.targetList)



	def strGrid(self,stateDescription): # reads description of the state as a dict and return the grid as a string
		currentPosition = (stateDescription['x'],stateDescription['y'])
		return fullGridStrPosition(self.walls, self.holes, self.targets, currentPosition)

	def getScore(self,stateDescription):
		x,y = stateDescription['x'],stateDescription['y']
		if self.targets[x][y]:
			return 1.0
		if self.holes[x][y]:
			return 0.0
		targetScore=1-normalizeDistance(self.targetDistance[x][y],self.maxTargetDistance)
		holeScore=normalizeDistance(self.holeDistance[x][y],self.maxHoleDistance)
		targetWt=9
		holeWt=1
		return normalizeFloat(targetWt*targetScore + holeWt*holeScore,0,targetWt+holeWt)

class MDPStateScoreDistance(MDPStateScoreInterface):

	def __init__(self,scoreFunction):
		self.scoreFunction = scoreFunction

	def getScore(self, executionEngine):
		endState = executionEngine.mdpEndState()
		stateDescription = executionEngine.mdpOperations.stateDescription(endState)
		return (self.scoreFunction(stateDescription))

def runGamesWithNN(stateStrFunction,prismFile,**kwargs):
	discount = kwargs['discount']
	kwargs.pop('discount')
	prismSimulator = prismToSimulator(prismFile)
	mdp = MDPOperations(prismSimulator,prismFile,stateStrFunction,discount)
	bitVector = prismSimulator._get_current_state()
	initState = MDPState(bitVector)
	labels = prismSimulator._report_labels()
	initPredicates = [MDPPredicate(label) for label in labels]
	traceEngine: MDPNNTraceEngine[MDPPredicate, MDPState, MDPAction, MDPStochasticAction] = MDPNNTraceEngine()
	results = traceEngine.runNNTrace(mdpState=initState, mdpPredicates=initPredicates, mdpOperations=mdp,**kwargs)

	return(results)

# def runGamesWithStorm(stateStrFunction,prismFile,**kwargs):
# 	discount = kwargs['discount']
# 	kwargs.pop('discount')
# 	prismSimulator = prismToSimulator(prismFile)
# 	mdp = MDPOperations(prismSimulator,prismFile,stateStrFunction,discount)
# 	bitVector = prismSimulator._get_current_state()
# 	initState = MDPState(bitVector)
# 	labels = prismSimulator._report_labels()
# 	initPredicates = [MDPPredicate(label) for label in labels]
# 	traceEngine: MDPStormTraceEngine[MDPPredicate, MDPState, MDPAction, MDPStochasticAction] = MDPStormTraceEngine()
# 	results = traceEngine.runStormTrace(mdpState=initState, mdpPredicates=initPredicates, mdpOperations=mdp,**kwargs)
#
# 	return(results)

def main():
	args = readCommand(sys.argv[1:])
	layoutFile = args['layout']
	prismFile = createPrismFile(layoutFile)
	stateStrFunction = frozenLake(layoutFile).strGrid
	# results = runGames(stateStrFunction,prismFile,**args)
	# engineList = [r[0] for r in results]
	# prettyConsole=True
	# runResults(engineList,quiet=prettyConsole,prettyConsole=prettyConsole)
	# layout = kwargs['layout']
	args.pop('layout')
	# mdp, initState, initPredicates = readFromFile(layout)
	isReplay = args['replay']
	args.pop('replay')
	if args['useMCTS']:
		args.pop('useMCTS')
		args.pop('useDT')
		args.pop('useNN')
		results = runGamesWithMCTS(stateStrFunction,prismFile,**args)
	elif args['useDT']:
		args.pop('useMCTS')
		args.pop('useDT')
		args.pop('useNN')
		raise Exception("decision tree not implemented")
		# runGamesWithDT(stateStrFunction,prismFile,**kwargs)
	elif args['useNN']:
		args.pop('useMCTS')
		args.pop('useDT')
		args.pop('useNN')
		results = runGamesWithNN(stateStrFunction,prismFile,**args)
	if isReplay:
		engineList = [r[0] for r in results]
		runResults(engineList,quiet=True,prettyConsole=True)

	return results

# def main():
# 	args = readCommand(sys.argv[1:])
# 	layoutFile = args['layout']
# 	prismFile = createPrismFile(layoutFile)
# 	stateStrFunction = frozenLake(layoutFile).strGrid
# 	results = runGames(stateStrFunction,prismFile,**args)
# 	engineList = [r[0] for r in results]
# 	prettyConsole=True
# 	runResults(engineList,quiet=prettyConsole,prettyConsole=prettyConsole)

if __name__ == "__main__":
	main()
