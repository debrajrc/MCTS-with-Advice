from simulationClasses import MDPPathAdviceInterface, MDPActionAdviceInterface, MDPStateScoreInterface, MDPExecutionEngine
from stormMdpClasses import MDPState, MDPOperations, MDPAction

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
	s = f"Time of the day: {stateDict['timeOfTheDay']} \t"
	s += f"Token: {stateDict['token']} \n"
	passengerList = []
	for i in range(0,2):
		if stateDict[f'c{i}_in'] == 1:
			passengerList.append(i) 
	s += f"Taxi : ({stateDict['xt']},{stateDict['yt']})\t (passengers: {passengerList})\n"
	for i in range(0,2):
		s += f"Client {i} : ({stateDict[f'xs_c{i}']},{stateDict[f'ys_c{i}']}) -- ({stateDict[f'xc_c{i}']},{stateDict[f'yc_c{i}']}) --> ({stateDict[f'xd_c{i}']},{stateDict[f'yd_c{i}']}) \t (remaining time: {stateDict[f'totalWaiting_c{i}']}) \n"
	return s