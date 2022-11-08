import glob, os, time, math, sys


CURRENTPWD = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(CURRENTPWD, '../src'))

from playFrozenLake import *

import adviceMCTS.util
import importlib

from adviceMCTS.util import NoMoveException

def loadConfig(pathName: str,fileName: str):
	sys.path.insert(0, pathName)
	config = importlib.import_module(fileName)
	return config

# LAYOUT_SET = 'Random10x10'
# # LAYOUT_SET = 'Test10x10'
# LAYOUTS_DIR = 'layouts'+os.sep+'frozenLake'+os.sep+LAYOUT_SET
# BENCHMARK_DIR = 'benchmarks'+os.sep+'frozenLake'+os.sep+LAYOUT_SET
#
# layouts = glob.glob(LAYOUTS_DIR+os.sep+"*.lay")

# optionRange=dict()

# optionRange["--numTraces"] = [1]
# optionRange["--steps"] = [100]
# optionRange["--numMCTSIters"] = [50]
# optionRange["--numMCTSSims"] = [10]
# optionRange["--horizon"] = [30]
# optionRange["--MCTSConstant"] = [0.7]
# optionRange["--stateScoreWeight"] = [1]
# optionRange["--MCTSPathAdviceSim"] = ["NonLoss"]

# sortedOptions=sorted(optionRange.items(),key=lambda x: x[0])
#
# maxIndex=1
# for k,v in sortedOptions:
#     maxIndex*=len(v)
#
# def indexToOptions(index):
#     index = int(index)
#     if index >= maxIndex or index < 0:
#         raise Exception ("bad index "+str(index))
#     options=[]
#     for k,v in sortedOptions:
#         options.append((k, v[index % len(v)]))
#         index=index // len(v)
#     if index != 0:
#         raise Exception("bad indexToOptions")
#     return options

def runBatch(index, experimentName, config, simulate=False):
	options, layoutBatch, numLayoutBatches = config.indexToOptions(index)
	li = 0
	layoutList = config.layouts
	stepSize = math.ceil(len(layoutList)/numLayoutBatches)
	beginIndex= layoutBatch*stepSize
	endIndex= beginIndex + stepSize
	if endIndex>len(layoutList):
		endIndex = len(layoutList)
	layoutBatchList = layoutList[beginIndex:endIndex]
	# print("layoutList:",layoutList)
	# print("layoutBatchList:",layoutBatchList)
	for l in layoutBatchList:
		DIR_NAME = config.BENCHMARK_DIR+os.sep+experimentName+os.sep+l[len(config.LAYOUTS_DIR+os.sep):-4]
		util.mkdir(DIR_NAME)
		if not simulate:
			f = open(DIR_NAME+os.sep+str(index)+'.ben', 'w+')
		command = "--layout "+l
		shortCommand = ""
		for k,v in options:
			command += " "+str(k)+" "+str(v)
			if k == "--numTraces":
				continue
			if k[0:2] == "--":
				kk = k[2:]
			shortCommand += " "+kk+" "+str(v)
		command += " --quietSim --quietMCTS"
		# print("command:",command)
		# print("shortCommand:",shortCommand)
		if not simulate:
			print(shortCommand, file= f)
		print(f"command {li+1}/{len(layoutBatchList)}:", command)
		args = readCommand(command.split(' '))
		startTime = time.time()
		try:
			if not simulate:
				results = runGames(**args)
			timeElapsed = time.time() - startTime
			if not simulate:
				print("timeElapsed:", timeElapsed, file= f)
				printResults(results, file = f)
				f.close()
		except NoMoveException:
			print('bad layout '+l)
		li+=1

def statFromEngineList(engineList):
	totalLength = 0
	totalScore = 0.0
	totalTerminal = 0
	for e in engineList:
		length = e.length()
		totalLength+=length
		score = e.mdpPathReward()
		totalScore+=score
		isTerminal = e.isTerminal()
		totalTerminal+=int(isTerminal)
		# print (f"length: {length}, score: {score}, isTerminal: {isTerminal}")
	avgLength = totalLength/len(engineList)
	avgScore = totalScore/len(engineList)
	avgTerminal = totalTerminal/len(engineList)
	print(f"average length: {avgLength}, score: {avgScore}, isTerminal: {avgTerminal}\n")
	return (totalLength, totalScore, totalTerminal, len(engineList))

def readBatch(smallIndex,experimentName,config,layoutName='**'):
	indexBatchList = config.getBatchList(smallIndex)
	files = list()
	for idx in indexBatchList:
		files += glob.glob(config.BENCHMARK_DIR+os.sep+experimentName+os.sep+layoutName+os.sep+str(idx)+".ben") + glob.glob(config.BENCHMARK_DIR+os.sep+experimentName+os.sep+'**'+os.sep+layoutName+os.sep+str(idx)+".ben")
	globalLength = 0
	globalScore = 0.0
	globalTerminal = 0
	globalNumTrace = 0
	globalTime = 0
	globalCommand = None
	for fileName in files:
		# print(fileName)
		f = open(fileName, 'r')
		command = f.readline().strip()
		if globalCommand is None:
			globalCommand = command
			print("\nreading results for index",indexBatchList,"command",command)
		elif command != globalCommand:
			print(command, globalCommand)
			raise Exception ("parsing error")
		timeL = f.readline().strip().split()
		if len(timeL) != 2:
			f.close()
			print("file", fileName, "empty")
			continue
		timeElapsed = float(timeL[1])
		engineList = readResults(f)
		f.close()

		print("file", fileName, "time:", timeElapsed, "numTraces:", len(engineList))
		# runResults(engineList,quiet=True)
		(totalLength, totalScore, totalTerminal, numTraces) = statFromEngineList(engineList)
		globalLength+=totalLength
		globalScore+=totalScore
		globalTerminal+=totalTerminal
		globalNumTrace+=numTraces
		globalTime+=timeElapsed
	if globalNumTrace == 0:
		return('-', 'NaN', 'NaN', 'NaN', 'NaN')
	avgLength = globalLength/globalNumTrace
	avgScore = globalScore/globalNumTrace
	avgTerminal = globalTerminal/globalNumTrace
	avgTime = globalTime/globalLength
	print(f"global for index {indexBatchList}: length: {globalLength}, time: {globalTime}")
	print(f"average for index {indexBatchList}: length: {avgLength}, score: {avgScore}, isTerminal: {avgTerminal}, timePerMove: {avgTime}")
	# return (totalLength, totalScore, totalTerminal, len(engineList))
	return(globalCommand, avgLength, avgScore, avgTerminal, avgTime)

def readAllBatches(experimentName, seperatelayouts, config):
	if seperatelayouts:
		lList =[l[len(config.LAYOUTS_DIR+os.sep):-4] for l in config.layouts]
	else:
		lList = ['**']
	f = open(config.BENCHMARK_DIR+os.sep+experimentName+os.sep+'benchmarkResult.csv','w+')
	print(','.join(['layoutName','cmd', 'index', 'avgLength', 'avgScore', 'avgTerminal', 'avgTime']),file=f)
	for layoutName in lList:
		for smallIndex in range(0, config.maxIndex // config.numLayoutBatches):
			cmd, avgLength, avgScore, avgTerminal, avgTime = readBatch(smallIndex,experimentName,config,layoutName)
			print (','.join([layoutName, cmd, str(smallIndex), str(avgLength), str(avgScore), str(avgTerminal), str(avgTime)]),file=f)
	f.close()

def main():
	if len(sys.argv) < 4:
		raise Exception("not enough args")
	cmd = sys.argv[1]
	expeName = sys.argv[2]
	pathName = 'experiments'+os.sep+'frozenLake'
	config = loadConfig(pathName,expeName)
	if cmd == 'run':
		index = int(sys.argv[3])
		runBatch(index, expeName, config)
	elif cmd == 'read':
		if sys.argv[3] == 'True':
			seperatelayouts = True
		elif sys.argv[3] == 'False':
			seperatelayouts = False
		else:
			raise Exception ('argument should be either True or False: ' + sys.argv[3])
		readAllBatches(expeName,seperatelayouts, config)
	elif cmd == 'sim':
		index = int(sys.argv[3])
		print("simulate run of index",index," out of [0,",config.maxIndex-1,"]")
		runBatch(index, expeName, config, simulate=True)
	elif cmd == 'maxIndex':
		index = int(sys.argv[3])
		print("goind to run index",index," out of [0,",config.maxIndex-1,"]")
	else:
		raise Exception ('argument should be either run or read or sim: ' + cmd)

# To run experiments: python3 run dataGenTest 0

if __name__ == '__main__':
	main()
