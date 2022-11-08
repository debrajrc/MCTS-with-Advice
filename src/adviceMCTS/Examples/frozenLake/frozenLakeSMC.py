import os
from adviceMCTS.Examples.frozenLake.frozenLakeCompare import *

def getValuePlots(layout, maxNumTraces):
	mdp, initState, initPredicates = readFromFile(layout)
	traceEngine = MDPDictTraceEngine()
	totalScore = 0
	totalWin = 0
	totalLoss = 0
	totalDraw = 0
	numGames = 0
	d = getStormDistDict(layout)
	for i in range(1,maxNumTraces):
		results = traceEngine.runDictTrace(numTraces = i, mdpState = initState, mdpPredicates = initPredicates, mdpOperations = mdp, d = d, horizonTrace = 1000, quietTrace = True, quietInfoStr = True, printEachStepTrace = False)
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
		print(i,avgScore,avgWin,avgLoss,avgDraw)

if __name__ == "__main__":
	layout = LAYOUTS_DIR + os.sep + "18_10x10_12.lay"
	maxNumTraces = 100
	getValuePlots(layout, maxNumTraces)
