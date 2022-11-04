
import sys, argparse, math, subprocess, os, yaml

# CURRENTPWD = os.path.dirname(os.path.abspath(__file__))

# sys.path.append(os.path.join(CURRENTPWD, '../src'))

from adviceMCTS.stormMdpClasses import *
from adviceMCTS.Examples.pacman.pacmanPrism import *
from tensorflow.keras import models
import json, joblib

def readCommand(argv):
	usageStr =  """
		USAGE:      python3 playPacmanStorm.py <options>

		EXAMPLE #1:	python3 pacman/playPacmanStorm.py --mcts --ActionScoreThreshold 0.9 --ActionScoreThresholdRoot 0.9 --MCTSActionAdvice Progress --MCTSActionAdviceRoot Progress --MCTSConstant 1000 --MCTSPathAdviceSim NonLoss --MCTSStateScore DistanceOld --ModelActionScore pacman/models/safetyScore/actiondata_all_5_8.h5 --TransformerActionScore None --ModelProgress pacman/models/iterMCTSData10/actiondata_I3_CTrue_10.h5 --TransformerProgress None --ModelActionScoreRoot pacman/models/safetyScore/actiondata_all_5_8.h5 --TransformerActionScoreRoot None --ModelProgressRoot pacman/models/iterMCTSData10/actiondata_I3_CTrue_10.h5 --TransformerProgressRoot None --horizon 10 --layout pacman/layouts/halfClassic.lay --numMCTSIters 4 --numMCTSSims 2 --numTraces 1 --stateScoreWeight 0.5 --steps 701 --ignoreNonDecisionStates -vvvvv --replay --replayDelay 0.05

		EXAMPLE #2:	python3 pacman/playPacmanStorm.py --nn --nnModel pacman/models/newMCTSData10/actiondata_all_5_10.h5 --nnTransformer None -l pacman/layouts/halfClassic.lay --replay -s 2100 --replayDelay 0.05
		"""

	parser = argparse.ArgumentParser(usageStr)

	# mode

	mode = parser.add_mutually_exclusive_group()
	mode.add_argument('--mcts', dest='useMCTS', action='store_true', default=False, help='Uses mcts')
	mode.add_argument('--nn', dest='useNN', action='store_true', default=False, help='Uses a neural network')
	mode.add_argument('--storm', dest='useStorm', action='store_true', default=False, help='Uses storm')
	mode.add_argument('--nnProgress', dest='useNNProgress', action='store_true', default=False, help='Uses two NN')
	mode.add_argument('--debug', dest='debug', action='store_true', default=False, help='Uses two NN to debug')

	parser.add_argument('-v', '--verbose', dest='verbose', action='count', default=0)
	parser.add_argument('-l', '-lay', '--layout', dest='layout', type=str, help='the LAYOUT', metavar='LAYOUT')
	parser.add_argument('-n', '-nTr', '--numTraces', dest='numTraces', type=int, help='the number of TRACES to play (Default: 1)', metavar='TRACES', default=1)
	parser.add_argument('-s', '-step', '--steps', dest='horizonTrace', type=int, help='the number of STEPS to play in each trace(Default: 50)', metavar='STEPS', default=50)
	parser.add_argument('--discount', dest='discount', type=float, help='Discount in MDP (Default: 1.0)', metavar='CONSTANT', default=1.0)

	# mcts specific arguments
	mctsParser = parser.add_argument_group('MCTS specific arguments')
	mctsParser.add_argument('-iter', '--numMCTSIters', dest='numMCTSIters', type=int, help='the number of ITERATIONS in MCTS(Default: 50)', metavar='ITERATIONS', default=50)
	mctsParser.add_argument('-sim', '--numMCTSSims', dest='numSims', type=int, help='the number of SIMULATIONS in each MCTS iterations(Default: 10)', metavar='SIMULATIONS', default=10)
	mctsParser.add_argument('-H', '--horizon', dest='horizon', type=int, help='HORIZON for MCTS (Default: 20)', metavar='HORIZON', default=20)
	mctsParser.add_argument('--ignoreNonDecisionStates', dest='ignoreNonDecisionStates', action='store_true', default=False, help='If set, states with a single action are ignored for horizon purposes')
	mctsParser.add_argument('-C', '--MCTSConstant', dest='mctsConstant', type=float, help='MCTS CONSTANT (write in float; Default: math.sqrt(2)/2)', metavar='CONSTANT')

	# selection advice
	mctsParser.add_argument('--MCTSActionAdvice', dest='mctsActionAdvice', type=str, help='MCTS Action ADVICE for selection(Default: Full)', metavar='ADVICE')
	mctsParser.add_argument('--ModelActionScore', dest='modelActionScore', type=str, help='MODEL file for Action Score during selection', metavar='MODEL')
	mctsParser.add_argument('--TransformerActionScore', dest='transformerActionScore', type=str, help='TRANSFORMER file for Action Score during selection', metavar='TRANSFORMER')
	mctsParser.add_argument('--ModelProgress', dest='modelProgress', type=str, help='Neural network MODEL file for progress', metavar='MODEL')
	mctsParser.add_argument('--TransformerProgress', dest='transformerProgress', type=str, help='Neural network TRANSFORMER file for progress', metavar='TRANSFORMER')
	mctsParser.add_argument('--ActionScoreThreshold', dest='actionScoreThreshold', type=float, help='THRESHOLD for actions while using NN or Storm as advice during selection (Default: 0.9)', metavar='THRESHOLD', default = 0.9)
	mctsParser.add_argument('--DepthActionScoreStorm', dest='DepthActionScoreStorm', type=int, help='DEPTH for Action Score during selection using Storm', metavar='DEPTH')

	# selection advice at root
	mctsParser.add_argument('--MCTSActionAdviceRoot', dest='mctsActionAdviceRoot', type=str, help='MCTS Action ADVICE for selection in the root node (Default: Full)', metavar='ADVICE')
	mctsParser.add_argument('--ModelActionScoreRoot', dest='modelActionScoreRoot', type=str, help='MODEL file for Action Score during selection in the root node', metavar='MODEL')
	mctsParser.add_argument('--TransformerActionScoreRoot', dest='transformerActionScoreRoot', type=str, help='MODEL file for Action Score during selection in the root node', metavar='TRANSFORMER')
	mctsParser.add_argument('--ModelProgressRoot', dest='nnModelProgressRoot', type=str, help='Neural network MODEL file for progress', metavar='MODEL')
	mctsParser.add_argument('--TransformerProgressRoot', dest='nnTransformerProgressRoot', type=str, help='Neural network TRANSFORMER file for progress', metavar='TRANSFORMER')
	mctsParser.add_argument('--ActionScoreThresholdRoot', dest='actionScoreThresholdRoot', type=float, help='THRESHOLD for actions while using NN as advice during selection in the root node (Default: 0.9)', metavar='THRESHOLD', default = 0.9)
	mctsParser.add_argument('--DepthActionScoreStormRoot', dest='DepthActionScoreStormRoot', type=int, help='DEPTH for Action Score during selection using Storm in the root node', metavar='DEPTH')

	# simulation advice
	mctsParser.add_argument('--MCTSActionAdviceSim', dest='mctsActionAdviceSim', type=str, help='MCTS Action ADVICE for simulation(Default: Full)', metavar='ADVICE')
	mctsParser.add_argument('--MCTSPathAdviceSim', dest='mctsPathAdviceSim', type=str, help='MCTS Path ADVICE for simulation(Default: Full)', metavar='ADVICE')
	mctsParser.add_argument('--ModelActionScoreSim', dest='modelActionScoreSim', type=str, help='MODEL file for Action Score during simulation', metavar='MODEL')
	mctsParser.add_argument('--TransformerActionScoreSim', dest='transformerActionScoreSim', type=str, help='TRANSFORMER file for Action Score during simulation', metavar='TRANSFORMER')
	mctsParser.add_argument('--ActionScoreThresSim', dest='actionScoreThresSim', type=float, help='THRESHOLD for actions while using NN or Storm as advice during simulation (Default: 0.9)', metavar='THRESHOLD', default = 0.9)

	# terminal score
	mctsParser.add_argument('--MCTSStateScore', dest='mctsStateScore', type=str, help='MCTS State SCORE for simulation (Default: Zero)', metavar='SCORE')
	mctsParser.add_argument('--StateScoreModel', dest='modelStateScore', type=str, help='MODEL file for State Score', metavar='MODEL')
	mctsParser.add_argument('--StateScoreModelExtra', dest='modelStateScoreExtra', type=str, help='Extra information about MODEL file for State Score', metavar='MODEL')
	mctsParser.add_argument('-a', '--stateScoreWeight', dest='alpha', type=float, help='MCTS State Score weight (write in float; Default: 0.5', metavar='CONSTANT', default=0.5)

	# nn specific arguments
	nnParser = parser.add_argument_group('NN specific arguments')
	nnParser.add_argument('--nnModel', dest='nnModel', type=str, help='Neural network MODEL file', metavar='MODEL')
	nnParser.add_argument('--nnTransformer', dest='nnTransformer', type=str, help='Neural network TRANSFORMER file', metavar='TRANSFORMER')
	nnParser.add_argument('--nnThreshold', dest='nnThreshold', type=float, help='THRESHOLD for actions while using NN (Default: 0.9)', metavar='THRESHOLD', default = 0.9)

	# nn (progress) specific arguments
	nnProgressParser = parser.add_argument_group('NN progress specific arguments')
	nnProgressParser.add_argument('--nnModelSafety', dest='nnModelSafety', type=str, help='Neural network MODEL file for safety', metavar='MODEL')
	nnProgressParser.add_argument('--nnTransformerSafety', dest='nnTransformerSafety', type=str, help='Neural network TRANSFORMER file for safety', metavar='TRANSFORMER')
	nnProgressParser.add_argument('--nnModelProgress', dest='nnModelProgress', type=str, help='Neural network MODEL file for progress', metavar='MODEL')
	nnProgressParser.add_argument('--nnTransformerProgress', dest='nnTransformerProgress', type=str, help='Neural network TRANSFORMER file for progress', metavar='TRANSFORMER')
	nnProgressParser.add_argument('--nnThresholdProgress', dest='nnThresholdProgress', type=float, help='THRESHOLD for actions while using NN (Default: 0.9)', metavar='THRESHOLD', default = 0.9)

	# storm specific arguments
	stormParser = parser.add_argument_group('Storm specific arguments')
	stormParser.add_argument('--stormDepth', dest='stormDepth', type=int, help='DEPTH for storm (Default: 3)', metavar='DEPTH', default = 3)
	stormParser.add_argument('--stormThreshold', dest='stormThreshold', type=float, help='THRESHOLD for actions while using Storm (Default: 0.9)', metavar='THRESHOLD', default = 0.9)

	# output specific arguments
	vParser = parser.add_argument_group('Verbosity during MCTS')
	vParser.add_argument('-qm', '--quietMCTS', dest='quietMCTS', action='store_true', default=None, help='Does not print console outputs with doing MCTS')
	vParser.add_argument('-qt', '--quietTrace', dest='quietTrace', action='store_true', default=None, help='Does not print the trace')
	vParser.add_argument('-qs', '--quietSim', dest='quietSim', action='store_true', default=None, help='Does not print the sims')
	vParser.add_argument('-ps', '--printEachStep', dest='printEachStep', action='store_true', default=None, help='Print each step of the sims')
	vParser.add_argument('-psT', '--printEachStepTrace', dest='printEachStepTrace', action='store_true', default=None,    help='Print each step of the trace')
	vParser.add_argument('-qi', '--quietInfostr', dest='quietInfoStr', action='store_true', default=False, help='Does not print additional information related to actions')

	# replay
	rParser = parser.add_argument_group('Other options')
	rParser.add_argument('-r', '--replay', dest='replay', action='store_true', default=False, help='Replays the game')
	rParser.add_argument('--replayDelay', dest='replayDelay', type=float, help='replay DELAY in seconds (Default: 0)', metavar='DELAY', default = 0)

	# config file (beta)
	parser.add_argument('--configFile', dest='configFile', type=str, help='Config FILE', metavar='FILE')

	options = parser.parse_args(argv)

	if options.configFile:
		with open(options.configFile) as f:
			configs = yaml.load(f, Loader=yaml.FullLoader)
		for key, value in configs.items():
			setattr(options, key, value)

	args = {}

	if options.layout is None:
		args['layout'] = 'layouts'+os.sep+'pacman'+os.sep+'halfClassic.lay'
	else:
		args['layout'] = options.layout

	# mcts specific arguments
	args['numTraces'] = options.numTraces
	args['horizonTrace'] = options.horizonTrace
	args['discount'] = options.discount

	# mode
	args['useMCTS'] = options.useMCTS
	args['useNN'] = options.useNN
	args['useStorm'] = options.useStorm
	args['useNNProgress'] = options.useNNProgress
	# args['debug'] = options.debug

	pacmanEngine = createEngine(args['layout'])
	args['pacmanEngine'] = pacmanEngine

	# replay
	args['replay'] = options.replay
	args['replayDelay'] = options.replayDelay

	# verbosity
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

	# mcts specific arguments
	if options.useMCTS:
		args['numMCTSIters'] = options.numMCTSIters
		args['numSims'] = options.numSims

		if options.mctsConstant is None:
			mctsConstant = math.sqrt(2)/2
		else:
			mctsConstant = options.mctsConstant

		# Terminal score
		if options.mctsStateScore is None:
			mdpStateScore=MDPStateScore()
		elif options.mctsStateScore == "Zero":
			mdpStateScore=MDPStateScoreZero()
		elif options.mctsStateScore == "Distance":
			scoreFunction = pacmanEngine.getScore
			mdpStateScore=MDPStateScoreDistance(scoreFunction)
		elif options.mctsStateScore == "DistanceOld":
			(X,Y,numGhosts,walls) = pacmanEngine.getLayoutDescription()
			mdpStateScore=MDPStateScoreDistanceOld(X,Y,numGhosts,walls)
		elif options.mctsStateScore == "mctsNN":
			modelFile = options.modelStateScore
			modelExtraFile = options.modelStateScoreExtra
			model = models.load_model(modelFile)
			extraData = np.load(modelExtraFile,allow_pickle=True)
			minY = extraData["YMin"]
			maxY = extraData["YMax"]
			mdpStateScore=MDPStateScoreMctsNN(pacmanEngine,model,minY,maxY)
		elif options.mctsStateScore == "safetyNN":
			modelFile = options.modelStateScore
			model = models.load_model(modelFile)
			mdpStateScore=MDPStateScoreSafetyNN(pacmanEngine,model)
		elif options.mctsStateScore == "NN":
			modelFile = options.modelStateScore
			model = models.load_model(modelFile)
			mdpStateScore=MDPStateScoreNN(model) # TODO: not implemented
		else:
			raise Exception ("Unknown state score")

		alpha=options.alpha

		# selection advice
		if options.mctsActionAdvice is None:
			mdpActionAdvice = MDPFullActionAdvice()

		elif options.mctsActionAdvice == "NN":
			modelFileSel = options.modelActionScore
			transformerFileSel = options.transformerActionScore
			thresholdSel = options.actionScoreThreshold
			modelSel = models.load_model(modelFile)
			if transformerFileSel == "None":
				transformerSel = None
			else:
				transformerSel = joblib.load(transformerFileSel)
			mdpActionAdvice = MDPNNActionAdvice(modelSel,transformerSel,pacmanEngine,thresholdSel)

		elif options.mctsActionAdvice == "Progress":
			modelFileSel = options.modelActionScore
			transformerFileSel = options.transformerActionScore
			thresholdSel = options.actionScoreThreshold
			modelSel = models.load_model(modelFileSel)
			if transformerFileSel == "None":
				transformerSel = None
			else:
				transformerSel = joblib.load(transformerFileSel)
			# mdpActionAdvice = MDPNNActionAdvice(modelSim,transformerSim,pacmanEngine,threshold)
			modelFileProgressSel = options.modelProgress
			transformerFileProgressSel = options.transformerProgress
			if options.modelProgress == "None":
				modelProgressSel = None
			else:
				modelProgressSel = models.load_model(modelFileProgressSel)
			if options.transformerProgress == "None":
				transformerProgressSel = None
			else:
				transformerProgressSel = joblib.load(transformerFileProgressSel)
			# args['threshold'] = options.nnThresholdProgress
			mdpActionAdvice = MDPNNActionAdviceProgress(modelSel,modelProgressSel,transformerSel,transformerProgressSel,pacmanEngine, thresholdSel)

		elif options.mctsActionAdvice == "Storm":
			depthSel = options.DepthActionScoreStorm
			thresholdSel = options.actionScoreThreshold
			mdpActionAdvice = MDPStormActionAdvice(depthSel,pacmanEngine,thresholdSel)

		elif options.mctsActionAdvice == "Full":
			mdpActionAdvice = MDPFullActionAdvice()
		elif options.mctsActionAdvice == "EXNonLoss":
			mdpActionAdvice = MDPEXNonLossActionAdvice()
		elif options.mctsActionAdvice == "AXNonLoss":
			mdpActionAdvice = MDPAXNonLossActionAdvice()
		else:
			raise Exception ("Unknown action advice for selection")

		# selection advice at root
		if options.mctsActionAdviceRoot is None:
			mdpActionAdviceRoot = MDPFullActionAdvice()

		elif options.mctsActionAdviceRoot == "NN":
			modelFileRoot = options.modelActionScoreRoot
			transformerFileRoot = options.transformerActionScoreRoot
			thresholdRoot = options.actionScoreThresholdRoot
			modelRoot = models.load_model(modelFileRoot)
			if transformerFileRoot == "None":
				transformerRoot = None
			else:
				transformerRoot = joblib.load(transformerFileRoot)
			mdpActionAdviceRoot = MDPNNActionAdvice(modelRoot,transformerRoot,pacmanEngine,thresholdRoot)

		elif options.mctsActionAdviceRoot == "Progress":
			modelFileRoot = options.modelActionScoreRoot
			transformerFileRoot = options.transformerActionScoreRoot
			thresholdRoot = options.actionScoreThreshold
			modelSafetyRoot = models.load_model(modelFileRoot)
			if transformerFileRoot == "None":
				transformerSafetyRoot = None
			else:
				transformerSafetyRoot = joblib.load(transformerFileRoot)
			modelFileProgressRoot = options.nnModelProgressRoot
			transformerFileProgressRoot = options.nnTransformerProgressRoot
			if options.nnModelProgressRoot == "None":
				modelProgressRoot = None
			else:
				modelProgressRoot = models.load_model(modelFileProgressRoot)
			if options.nnTransformerProgressRoot == "None":
				transformerProgressRoot = None
			else:
				transformerProgressRoot = joblib.load(transformerFileProgressRoot)
			# args['threshold'] = options.nnThresholdProgress
			mdpActionAdviceRoot = MDPNNActionAdviceProgress(modelSafetyRoot,modelProgressRoot,transformerSafetyRoot,transformerProgressRoot,pacmanEngine,thresholdRoot)

		elif options.mctsActionAdviceRoot == "Storm":
			depthRoot = options.DepthActionScoreStormRoot
			thresholdRoot = options.actionScoreThresholdRoot
			mdpActionAdviceRoot = MDPStormActionAdvice(depthRoot,pacmanEngine,thresholdRoot)

		elif options.mctsActionAdviceRoot == "Full":
			mdpActionAdviceRoot = MDPFullActionAdvice()
		elif options.mctsActionAdviceRoot == "EXNonLoss":
			mdpActionAdviceRoot = MDPEXNonLossActionAdvice()
		elif options.mctsActionAdviceRoot == "AXNonLoss":
			mdpActionAdviceRoot = MDPAXNonLossActionAdvice()
		else:
			raise Exception ("Unknown action advice for selection at root")

		# simulation advice
		if options.mctsActionAdviceSim is None:
			mdpActionAdviceSim = MDPFullActionAdvice()

		elif options.mctsActionAdviceSim == "NN":
			modelFileSim = options.modelActionScoreSim
			transformerFileSim = options.transformerActionScoreSim
			thresholdSim = options.actionScoreThresSim
			modelSim = models.load_model(modelFileSim)
			if transformerFileSim == "None":
				transformerSim = None
			else:
				transformerSim = joblib.load(transformerFileSim)
			mdpActionAdviceSim = MDPNNActionAdvice(modelSim,transformerSim,pacmanEngine,thresholdSim)

		elif options.mctsActionAdviceSim == "Full":
			mdpActionAdviceSim = MDPFullActionAdvice()
		elif options.mctsActionAdviceSim == "EXNonLoss":
			mdpActionAdviceSim = MDPEXNonLossActionAdvice()
		elif options.mctsActionAdviceSim == "AXNonLoss":
			mdpActionAdviceSim = MDPAXNonLossActionAdvice()
		else:
			raise Exception ("Unknown action advice for simulation")

		# path advice
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

	elif options.useNN:
		modelFile = options.nnModel
		transformerFile = options.nnTransformer
		if options.nnModel == "None":
			model = None
		else:
			model = models.load_model(modelFile)
		if options.nnTransformer == "None":
			transformer = None
		else:
			transformer = joblib.load(transformerFile)
		# args['threshold'] = options.nnThreshold
		args['uniformAdvice'] = MDPNNActionAdvice(model, transformer, pacmanEngine, options.nnThreshold)

	elif options.useNNProgress:
		modelFileSafety = options.nnModelSafety
		transformerFileSafety = options.nnTransformerSafety
		if options.nnModelSafety == "None":
			modelSafety = None
		else:
			modelSafety = models.load_model(modelFileSafety)
		if options.nnTransformerSafety == "None":
			transformerSafety = None
		else:
			transformerSafety = joblib.load(transformerFileSafety)
		modelFileProgress = options.nnModelProgress
		transformerFileProgress = options.nnTransformerProgress
		if options.nnModelProgress == "None":
			modelProgress = None
		else:
			modelProgress = models.load_model(modelFileProgress)
		if options.nnTransformerProgress == "None":
			transformerProgress = None
		else:
			transformerProgress = joblib.load(transformerFileProgress)
		# args['threshold'] = options.nnThresholdProgress
		args['uniformAdvice'] = MDPNNActionAdviceProgress(modelSafety,modelProgress,transformerSafety,transformerProgress,pacmanEngine,options.nnThresholdProgress)

	elif options.useStorm:
		# args['depth'] = options.stormDepth
		# args['threshold'] = options.stormThreshold
		args['uniformAdvice'] = MDPStormActionAdvice(options.stormDepth,pacmanEngine,options.stormThreshold)

	elif options.debug:
		modelFileSafety = options.nnModelSafety
		transformerFileSafety = options.nnTransformerSafety
		if options.nnModelSafety == "None":
			modelSafety = None
		else:
			modelSafety = models.load_model(modelFileSafety)
		if options.nnTransformerSafety == "None":
			transformerSafety = None
		else:
			transformerSafety = joblib.load(transformerFileSafety)
		modelFileProgress = options.nnModelProgress
		transformerFileProgress = options.nnTransformerProgress
		if options.nnModelProgress == "None":
			modelProgress = None
		else:
			modelProgress = models.load_model(modelFileProgress)
		if options.nnTransformerProgress == "None":
			transformerProgress = None
		else:
			transformerProgress = joblib.load(transformerFileProgress)
		# args['threshold'] = options.nnThresholdProgress
		advice1 = MDPNNActionAdviceProgress(modelSafety,modelProgress,transformerSafety,transformerProgress,pacmanEngine,options.nnThreshold)
		advice2 = MDPNNActionAdvice(modelSafety, transformerSafety, pacmanEngine, options.nnThreshold)
		args['uniformAdvice'] = MDPDebugActionAdvice(advice1,advice2)

	else:
		raise SystemExit(": error: use one of these arguments to select the mode: [--mcts | --nn | --storm | -- nnProgress]")

	return (args)

def replayResults(engineList, layoutFile, delay):
	X,Y,numGhosts,layoutText,agentInfo = readFromFile(layoutFile)
	p = pacmanEngine(X,Y,numGhosts,layoutText,agentInfo)
	stateStrFunction = p.printLayout
	for e in engineList:
		e.mdpOperations.updateStrFunction(stateStrFunction)
	runResults(engineList,cursesDelay=delay, quiet=True,prettyConsole=True)

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

def runGamesWithNNProgress(stateStrFunction,prismFile,**kwargs):
	discount = kwargs['discount']
	kwargs.pop('discount')
	prismSimulator = prismToSimulator(prismFile)
	mdp = MDPOperations(prismSimulator,prismFile,stateStrFunction,discount)
	bitVector = prismSimulator._get_current_state()
	initState = MDPState(bitVector)
	labels = prismSimulator._report_labels()
	initPredicates = [MDPPredicate(label) for label in labels]
	traceEngine: MDPNNTraceEngineProgress[MDPPredicate, MDPState, MDPAction, MDPStochasticAction] = MDPNNTraceEngineProgress()
	results = traceEngine.runNNTrace(mdpState=initState, mdpPredicates=initPredicates, mdpOperations=mdp,**kwargs)

	return(results)

def runGamesWithStorm(stateStrFunction,prismFile,**kwargs):
	discount = kwargs['discount']
	kwargs.pop('discount')
	prismSimulator = prismToSimulator(prismFile)
	mdp = MDPOperations(prismSimulator,prismFile,stateStrFunction,discount)
	bitVector = prismSimulator._get_current_state()
	initState = MDPState(bitVector)
	labels = prismSimulator._report_labels()
	initPredicates = [MDPPredicate(label) for label in labels]
	traceEngine: MDPStormTraceEngine[MDPPredicate, MDPState, MDPAction, MDPStochasticAction] = MDPStormTraceEngine()
	results = traceEngine.runStormTrace(mdpState=initState, mdpPredicates=initPredicates, mdpOperations=mdp,**kwargs)

	return(results)

def runGamesWithUniformAdvice(stateStrFunction,prismFile,**kwargs):
	discount = kwargs['discount']
	kwargs.pop('discount')
	prismSimulator = prismToSimulator(prismFile)
	mdp = MDPOperations(prismSimulator,prismFile,stateStrFunction,discount)
	bitVector = prismSimulator._get_current_state()
	initState = MDPState(bitVector)
	labels = prismSimulator._report_labels()
	initPredicates = [MDPPredicate(label) for label in labels]
	traceEngine: MDPUniformAdviceTraceEngine[MDPPredicate, MDPState, MDPAction, MDPStochasticAction] = MDPUniformAdviceTraceEngine()
	results = traceEngine.runTrace(mdpState=initState, mdpPredicates=initPredicates, mdpOperations=mdp,**kwargs)

	return(results)

def runGamesFromCommand(command):
	args = readCommand(command)
	layoutFile = args['layout']
	pacmanEngine = args['pacmanEngine']
	prismFile = pacmanEngine.createPrismFile()
	stateStrFunction = pacmanEngine.printLayout
	args.pop('layout')
	isReplay = args['replay']
	args.pop('replay')
	cursesDelay = args['replayDelay']
	args.pop('replayDelay')
	if args['useMCTS']:
		args.pop('useMCTS')
		args.pop('useNN')
		args.pop('useStorm')
		args.pop('useNNProgress')
		args.pop('pacmanEngine')
		results = runGamesWithMCTS(stateStrFunction,prismFile,**args)
	else:
		# elif args['useNN']:
		# 	args.pop('useMCTS')
		# 	args.pop('useNN')
		# 	args.pop('useStorm')
		# 	args.pop('useNNProgress')
		# 	results = runGamesWithNN(stateStrFunction,prismFile,**args)
		# elif args['useNNProgress']:
		# 	args.pop('useMCTS')
		# 	args.pop('useNN')
		# 	args.pop('useStorm')
		# 	args.pop('useNNProgress')
		# 	results = runGamesWithNNProgress(stateStrFunction,prismFile,**args)
		# elif args['useStorm']:
		if not args['useStorm']:
			args.pop('pacmanEngine')
		args.pop('useMCTS')
		args.pop('useNN')
		args.pop('useStorm')
		args.pop('useNNProgress')
		# results = runGamesWithStorm(stateStrFunction,prismFile,**args)
		results = runGamesWithUniformAdvice(stateStrFunction,prismFile,**args)
	if isReplay:
		engineList = [r[0] for r in results]
		prettyConsole=True
		runResults(engineList,cursesDelay=cursesDelay,quiet=prettyConsole,prettyConsole=prettyConsole)
	return results

if __name__ == "__main__":
	runGamesFromCommand(sys.argv[1:])
