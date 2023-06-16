import os, sys
from simulationClasses import MDPPathAdviceInterface, MDPActionAdviceInterface, MDPStateScoreInterface, MDPExecutionEngine
from stormMdpClasses import MDPState, MDPOperations, MDPAction

dirname = os.path.dirname(__file__)

sys.path.append(dirname)
from pacmanPrism import pacmanEngine

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
	p = pacmanEngine.fromFile(dirname+"/layouts/halfClassic.lay")
	return p.printLayout(stateDict)
