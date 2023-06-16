import sys, argparse, os, yaml, importlib
from stormMdpClasses import *

def readFromYaml(yamlFile):
	with open(yamlFile, 'r') as stream:
		try:
			return yaml.safe_load(stream)
		except yaml.YAMLError as exc:
			print(exc)

def readParameters(yamlFile):
	params = readFromYaml(yamlFile)
	args = {}

	# verbosity
	verbosity = int(params["verbosity"])
	if verbosity == 0:
		quietMCTS=True
		quietTrace=True
		quietSim=True
		printEachStep=False
		printEachStepTrace=False
	elif verbosity == 1:
		quietMCTS=True
		quietTrace=False
		quietSim=True
		printEachStep=False
		printEachStepTrace=False
	elif verbosity == 2:
		quietMCTS=True
		quietTrace=False
		quietSim=True
		printEachStep=False
		printEachStepTrace=True
	elif verbosity == 3:
		quietMCTS=False
		quietTrace=False
		quietSim=True
		printEachStep=False
		printEachStepTrace=True
	elif verbosity == 4:
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

	args["quietTrace"] = quietTrace
	args["quietInfoStr"] = quietMCTS
	args["printEachStepTrace"] = printEachStepTrace

	# prism file location where we write additional methods
	prismFile = params["prism file"].replace(".py","")

	# basic parameters
	args["numTraces"] = int(params["number of games"])
	args["horizonTrace"] = int(params["number of steps"])
	args["discount"] = float(params["discount factor"])

	# mcts parameters
	args["numSims"] = int(params["mcts"]["number of simulations"])
	args["numMCTSIters"] = int(params["mcts"]["number of iterations"])
	horizon = int(params["mcts"]["horizon"])
	mctsConstant = float(params["mcts"]["mcts constant"])
	alpha = float(params["mcts"]["alpha"])

	# mdp parameters
	pythonFile = params["other parameters"]["python file"]
	module = importlib.import_module(pythonFile)
	otherParameters = params["other parameters"]

	# Terminal score
	if "state score" in otherParameters:
		mdpStateScoreFunction = otherParameters["state score"]
		mdpStateScore = getattr(module, mdpStateScoreFunction)()
	else:
		mdpStateScore = MDPStateScore()

	# selection advice
	if "selection advice" in otherParameters:
		mdpActionAdviceFunction = otherParameters["selection advice"]
		mdpActionAdvice = getattr(module, mdpActionAdviceFunction)()
	else:
		mdpActionAdvice = MDPSafeActionAdvice()
	if "selection advice root" in otherParameters:
		mdpActionAdviceRootFunction = otherParameters["selection advice at root"]
		mdpActionAdviceRoot = getattr(module, mdpActionAdviceRootFunction)()
	else:
		mdpActionAdviceRoot = MDPSafeActionAdvice()

	# simulation advice
	if "simulation action advice" in otherParameters:
		mdpActionAdviceSimFunction = otherParameters["simulation action advice"]
		mdpActionAdviceSim = getattr(module, mdpActionAdviceSimFunction)()
	else:
		mdpActionAdviceSim = MDPSafeActionAdvice()
	if "simulation path advice" in otherParameters:
		mdpPathAdviceSimFunction = otherParameters["simulation path advice"]
		mdpPathAdviceSim = getattr(module, mdpPathAdviceSimFunction)()
	else:
		mdpPathAdviceSim = MDPNonLossPathAdvice()

	# some other parameters
	ignoreNonDecisionStates = True
	mdpActionStrategy = MDPUniformActionStrategy()
	rejectFactor = 10
	quiet = quietSim
	printCompact = True

	optionsSimulationEngine=OptionsSimulationEngine(horizon=horizon, ignoreNonDecisionStates = ignoreNonDecisionStates, mdpActionStrategy=mdpActionStrategy, mdpActionAdvice=mdpActionAdviceSim, mdpPathAdvice=mdpPathAdviceSim, mdpStateScore=mdpStateScore, alpha=alpha, rejectFactor=rejectFactor, quiet=quiet, quietInfoStr=printEachStepTrace, printEachStep=printEachStep, printCompact=printCompact)

	optionsMCTSEngine=OptionsMCTSEngine(horizon=horizon, ignoreNonDecisionStates = ignoreNonDecisionStates, mctsConstant=mctsConstant, mdpActionStrategy=mdpActionStrategy, mdpActionAdvice=mdpActionAdvice, mdpActionAdviceRoot=mdpActionAdviceRoot, optionsSimulationEngine=optionsSimulationEngine, mdpThresholdActionAdvice = None, mdpThresholdPathAdvice = None, quiet=quietMCTS, quietInfoStr=quietMCTS)

	args['optionsMCTSEngine'] = optionsMCTSEngine

	# a function to print the states nice
	if "print function" in otherParameters:
		niceStrFunction = otherParameters["print function"]
		niceStr = getattr(module, niceStrFunction)
	else:
		niceStr = str

	return prismFile, niceStr, args

def main():
	prismFile, niceStr, args = readParameters(sys.argv[1])
	results = runGamesWithMCTS(niceStr,prismFile,**args)
	engineList = [r[0] for r in results]
	prettyConsole=True
	cursesDelay=0.1
	runResults(engineList,cursesDelay=cursesDelay,quiet=prettyConsole,prettyConsole=prettyConsole)

if __name__ == "__main__":
	main()