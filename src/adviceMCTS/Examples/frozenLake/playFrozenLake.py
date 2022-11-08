# playFrozenLake.py

import sys, argparse, math, os, pickle

CURRENTPWD = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(CURRENTPWD, '../src'))

from adviceMCTSmdpClasses import *
from adviceMCTS.Examples.frozenLake.frozenLakeMdpClasses import *
from adviceMCTS.simulationClasses import *
from adviceMCTS.Examples.frozenLake.frozenLake import *

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
	mode.add_argument('--multiNN', dest='useMultiNN', action='store_true', default=False, help='Uses 2 neural network')
	mode.add_argument('--storm', dest='useStorm', action='store_true', default=False, help='Uses storm')
	mode.add_argument('--stormDist', dest='useStormDist', action='store_true', default=False, help='Uses storm and minimizes distance')

	parser.add_argument('-v', '--verbose', dest='verbose', action='count', default=0)
	parser.add_argument('-l', '-lay', '--layout', dest='layout', type=str, help='the LAYOUT', metavar='LAYOUT')
	parser.add_argument('-n', '-nTr', '--numTraces', dest='numTraces', type=int, help='the number of TRACES to play (Default: 1)', metavar='TRACES', default=1)
	parser.add_argument('-s', '-step', '--steps', dest='horizonTrace', type=int, help='the number of STEPS to play in each trace(Default: 50)', metavar='STEPS', default=50)
	parser.add_argument('--decisionTree', dest='decisionTree', type=str, help='decision TREE file for actions', metavar='TREE')
	parser.add_argument('--nnModel', dest='nnModel', type=str, help='neural network MODEL file for actions', metavar='MODEL')
	parser.add_argument('--nnThreshold', dest='nnThreshold', type=float, help='THRESHOLD to choose actions using neural network MODEL file for actions', metavar='THRESHOLD')
	parser.add_argument('--nnModel1', dest='nnModel1', type=str, help='1st neural network MODEL file for actions', metavar='MODEL')
	parser.add_argument('--nnThreshold1', dest='nnThreshold1', type=float, help='THRESHOLD to choose actions using 1st neural network MODEL file for actions', metavar='THRESHOLD')
	parser.add_argument('--nnModel2', dest='nnModel2', type=str, help='2nd neural network MODEL file for actions', metavar='MODEL')
	parser.add_argument('--nnThreshold2', dest='nnThreshold2', type=float, help='THRESHOLD to choose actions using 2nd neural network MODEL file for actions', metavar='THRESHOLD')
	parser.add_argument('--stormThreshold', dest='stormThreshold', type=float, help='THRESHOLD to choose actions using storm for actions', metavar='THRESHOLD')
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
	args['useMCTS'] = options.useMCTS
	args['useDT'] = options.useDT
	args['useNN'] = options.useNN
	args['useMultiNN'] = options.useMultiNN
	args['useStorm'] = options.useStorm
	args['useStormDist'] = options.useStormDist

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
			mdpStateScore=MDPStateScoreDistance()
		elif options.mctsStateScore == "NN":
			from tensorflow import keras
			modelFile = options.modelStateScore
			layout = args['layout']
			model = keras.models.load_model(modelFile,compile = False)
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
		model = keras.models.load_model(modelFile,compile = False)
		args['model'] = model
		args['threshold'] = options.nnThreshold
	elif options.useMultiNN:
		from tensorflow import keras
		modelFile1 = options.nnModel1
		model1 = keras.models.load_model(modelFile1,compile = False)
		args['model1'] = model1
		args['threshold1'] = options.nnThreshold1
		modelFile2 = options.nnModel2
		model2 = keras.models.load_model(modelFile2,compile = False)
		args['model2'] = model2
		args['threshold2'] = options.nnThreshold2
	elif options.useStorm:
		args['threshold'] = options.stormThreshold
	elif options.useStormDist:
		pass
	else:
		raise SystemExit(": error: use one of these arguments to select the mode: [--mcts | --dt | --nn | multiNN | stormDist]")

	args['replay'] = options.replay

	return (args)

def main():
	args = readCommand(sys.argv[1:])
	results = runGames(**args)

if __name__ == "__main__":
	main()
