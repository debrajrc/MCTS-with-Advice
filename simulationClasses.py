# simulationClasses.py

import math
import random, sys, time, curses

from typing import TypeVar, Type, Any, Optional, Sequence, List, Tuple, Dict, Union, Generic, NoReturn

import util
from util import raiseNotDefined, NoMoveException
from mdpClasses import *

##
# Class used to run an execution on a given MDP
class MDPExecutionEngine(Generic[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction]):
	TMDPOperations = TypeVar("TMDPOperations",bound=MDPOperationsInterface[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction])
	# Class used to run an execution on a given MDP

	# abstract methods that can be redefined if needed
	def __init__(self, mdpOperations: TMDPOperations, mdpExecution: MDPExecution[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction], nonDecisionLength: int) -> None:
		self.mdpOperations = mdpOperations # an MDPOperations instance
		self.mdpExecution = mdpExecution # an MDPExecution instance
		self.nonDecisionLength = nonDecisionLength # counts only non-decision states in the path length
	def deepCopy(self) -> "MDPExecutionEngine[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction]":
		return MDPExecutionEngine(self.mdpOperations.deepCopy(),self.mdpExecution.deepCopy(),self.nonDecisionLength)
	def __str__(self) -> str:
		return "(mdpOperations:"+str(self.mdpOperations)+", mdpExecution:"+str(self.mdpExecution) + ")"
	def __getitem__(self, item):
		return self.mdpExecution[item]
	def consoleStr(self) -> str:
		return self.mdpOperations.consoleStr()+"\n"+self.mdpExecution.consoleStr()
	def executionCopy(self) -> "MDPExecutionEngine[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction]":
		"""!
		half-shallow copy of an MDPExecutionEngine instance. Can be used to run independent executions on the same MDP.
		"""
		return MDPExecutionEngine(self.mdpOperations,self.mdpExecution.executionCopy(),self.nonDecisionLength)

	def append(self, mdpTransition: MDPTransition[TMDPAction, TMDPStochasticAction], nonDecisionAction : bool) -> None:
		# Appends a transition to an execution, and updates it
		# Redefine this method if you want something else than discounted total reward
		# If nonDecisionAction, nonDecisionLength does not increase
		mdpReward = 0.0
		isTerminal = self.mdpExecution.isTerminal
		if isTerminal:
			raise Exception("applying transition to terminal execution")
		mdpState = self.mdpExecution.mdpEndState
		# make changes to mdpState according to mdpTransition, and set mdpReward accordingly
		mdpReward = self.mdpOperations.applyTransitionOnState(mdpState, mdpTransition)
		# update mdpExecution
		# mdpOperations: TMDPOperations = seld.mdpOperations
		# d : float = mdpOperations.discountFactor
		self.mdpExecution.discountFactor *= self.mdpOperations.discountFactor
		self.mdpExecution.mdpPathReward += mdpReward * self.mdpExecution.discountFactor
		mdpPredicates: List[TMDPPredicate] = self.mdpOperations.getPredicates(mdpState)
		self.mdpExecution.append(mdpTransition, mdpPredicates)
		if not nonDecisionAction:
			self.nonDecisionLength += 1
		isTerminal = self.mdpOperations.isExecutionTerminal(self.mdpExecution)
		self.mdpExecution.isTerminal = isTerminal
		if isTerminal:
			self.mdpExecution.mdpPathReward += self.mdpOperations.getTerminalReward(self.mdpExecution)

	def getAllPredicates(self) -> List[TMDPPredicate]:
		return self.mdpOperations.getAllPredicates()
	def getPredicatesSequence(self) -> List[List[TMDPPredicate]]:
		return self.mdpExecution.mdpPath.mdpPredicatesSequence
	def appendPath(self, mdpPath: MDPPath[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction]) -> None:
		for i in range(mdpPath.length()):
			self.append(mdpPath.mdpTransitionsSequence[i])

	def fastReset(self, fastResetData: Tuple[int, TMDPState, float, bool, float]) -> None:
		fastResetDataExecution,nonDecisionLength=fastResetData
		self.mdpExecution.fastReset(fastResetDataExecution)
		self.nonDecisionLength = nonDecisionLength
	def getFastResetData(self) -> Tuple[int, TMDPState, float, bool, float]:
		return ((self.mdpExecution.getFastResetData(),self.nonDecisionLength))

	# methods that can be inherited
	def lastConsoleStr(self) -> str:
		return self.mdpExecution.lastConsoleStr()
	def suffixConsoleStr(self, depth: int) -> str:
		return self.mdpExecution.suffixConsoleStr(depth)
	def stateConsoleStr(self) -> str:
		return self.mdpExecution.stateConsoleStr()

	def length(self,ignoreNonDecisionStates) -> int:
		if ignoreNonDecisionStates:
			return self.nonDecisionLength
		else:
			return self.mdpExecution.length()
	def mdpPath(self) -> MDPPath[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction]:
		return self.mdpExecution.mdpPath
	def mdpEndState(self) -> TMDPState:
		return self.mdpExecution.mdpEndState
	def mdpPathReward(self) -> float:
		return self.mdpExecution.mdpPathReward
	def result(self) -> List[TMDPPredicate]:
		return(self.mdpExecution.mdpPath.mdpPredicatesSequence[-1])
	def isTerminal(self) -> bool:
		return self.mdpExecution.isTerminal
	def drawTransition(self, mdpAction: TMDPAction, quietInfoStr: bool) -> MDPTransition[TMDPAction, TMDPStochasticAction]:
		mdpStochasticAction: TMDPStochasticAction = self.mdpOperations.drawStochasticAction(self.mdpExecution.mdpEndState,mdpAction, quietInfoStr)
		return MDPTransition(mdpAction,mdpStochasticAction)


class OptionsReplayEngine():
	def __init__(self, quiet: bool, printEachStep: bool, printCompact: bool, cursesScr: "Optional[curses._CursesWindow]", cursesDelay: float) -> None:
		self.quiet = quiet
		self.printEachStep = printEachStep
		self.printCompact = printCompact
		self.cursesScr = cursesScr
		self.cursesDelay = cursesDelay
		if not cursesScr is None:
			if not quiet:
				raise Exception("should be quiet when using prettyConsole")
	def deepCopy(self) -> "OptionsSimulationEngine[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction]":
		return OptionsReplayEngine(quiet = self.quiet, printEachStep=self.printEachStep, printCompact=self.printCompact, cursesScr=self.cursesScr, cursesDelay=self.cursesDelay)
##
# MDPReplayEngine is used to replay an execution following a fixed path.
class MDPReplayEngine(Generic[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction]):
	TMDPOperations = TypeVar("TMDPOperations",bound=MDPOperationsInterface[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction])
	def __init__(self, mdpOperations: TMDPOperations, mdpReplayPath: MDPPath[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction], options: OptionsReplayEngine) -> None:
		isTerminal = (mdpReplayPath.length()==0)
		initState = mdpReplayPath.mdpInitialState.deepCopy()
		predicatesSequence = [ [ p.deepCopy() for p in mdpReplayPath.mdpPredicatesSequence[0] ] ]
		endState = initState.deepCopy()
		mdpPath: MDPPath[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction] = MDPPath(initState,[],predicatesSequence)
		mdpExecution: MDPExecution[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction] = MDPExecution(mdpPath,endState,0,isTerminal,1)

		self.mdpExecutionEngine: MDPExecutionEngine[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction] = MDPExecutionEngine(mdpOperations,mdpExecution,0) # an ExecutionEngine instance used to replay the decisions of mdpPath
		self.mdpPath = mdpReplayPath # an MDPPath instance

		self.quiet = options.quiet
		self.printEachStep = options.printEachStep
		self.printCompact = options.printCompact
		self.cursesScr = options.cursesScr
		self.cursesDelay = options.cursesDelay

		self.fastResetData=self.mdpExecutionEngine.getFastResetData()

	def __getitem__(self, item):
		return self.mdpPath[item]

	def _getMDPTransition(self) -> MDPTransition[TMDPAction, TMDPStochasticAction]:
		index = self.mdpExecutionEngine.length(ignoreNonDecisionStates=False)
		if index>=self.mdpPath.length():
			print(self.mdpExecutionEngine,"\n",self.mdpPath)
			raise Exception("index error, path too short")
		return self.mdpPath.mdpTransitionsSequence[index]

	def _extend(self, mdpTransition: MDPTransition[TMDPAction, TMDPStochasticAction]) -> None:
		"""!
		Simulates one time step.
		"""
		self.mdpExecutionEngine.append(mdpTransition,nonDecisionAction=True)

	def _runReplay(self, timeI: int) -> None:
		"""!
		Simulates one full run until a given depth.
		"""
		# if not self.quiet: print("--------------------------------------------------------------")
		# if not self.quiet: print("replaying for",depth-timeI,"steps from state",self.mdpExecutionEngine.stateConsoleStr())

		for j in range(timeI, timeI+1):
			if self.mdpExecutionEngine.isTerminal():
				break
			mdpTransition: MDPTransition[TMDPAction, TMDPStochasticAction] = self._getMDPTransition()
			self._extend(mdpTransition)
			if not self.quiet and self.printEachStep : print("step",j-timeI,":",self.mdpExecutionEngine.lastConsoleStr())
			if not self.cursesScr is None:
				self.cursesScr.clear()
				self.cursesScr.addstr(0,0,self.mdpExecutionEngine.mdpOperations.replayConsoleStr(self.mdpEndState())+"\n"+mdpTransition.consoleStr()+"\n"+self.mdpExecutionEngine.stateConsoleStr())
				self.cursesScr.refresh()
				time.sleep(self.cursesDelay)

	def advanceReplay(self) -> None:
		timeI=self.mdpExecutionEngine.length(ignoreNonDecisionStates = False)
		depth=timeI+1
		if depth > self.mdpPath.length():
			raise Exception("trying to advance finished replay")
		if not self.mdpExecutionEngine.isTerminal():
			# if not self.quiet: print("running replay for depth",timeI,"to",depth,"from",self.mdpExecutionEngine.stateConsoleStr())
			self._runReplay(timeI)
			# mdpPathReward=self.mdpExecutionEngine.mdpPathReward()
			if not self.quiet and not self.printEachStep:
				if not self.printCompact:
					print("|\t",self.mdpExecutionEngine.suffixConsoleStr(timeI))
				else:
					print("|\t",self.mdpExecutionEngine.length(ignoreNonDecisionStates = False),"steps to",self.mdpExecutionEngine.stateConsoleStr())
		else:
			raise Exception("trying to advance terminal replay")
			# if not self.quiet and not printEachStep: print("|\t",self.mdpExecutionEngine.suffixConsoleStr(timeI))

	def mdpEndState(self) -> TMDPState:
		return self.mdpExecutionEngine.mdpEndState()
	def getPathReward(self) -> float:
		return self.mdpExecutionEngine.mdpPathReward()
	def isTerminal(self) -> bool:
		return self.mdpExecutionEngine.isTerminal() or (self.mdpExecutionEngine.length(ignoreNonDecisionStates = False)>=self.mdpPath.length())

	def extractIndexedStateList(self, ignoreNonDecisionStates=False) -> List[Tuple[int,TMDPState]]:
		stateList = []
		endState = self.mdpEndState()
		if not ignoreNonDecisionStates or len(self.mdpExecutionEngine.mdpOperations.getLegalActions(endState)) > 1:
			stateList.append((self.mdpExecutionEngine.length(False),endState.deepCopy()))
		while not self.isTerminal():
			self.advanceReplay()
			endState = self.mdpEndState()
			if not ignoreNonDecisionStates or len(self.mdpExecutionEngine.mdpOperations.getLegalActions(endState)) > 1:
				stateList.append((self.mdpExecutionEngine.length(False),endState.deepCopy()))
		return stateList

	def resetReplay(self) -> None:
		self.mdpExecutionEngine.fastReset(self.fastResetData)


class MDPActionStrategyInterface(Generic[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction]):
	"""!
	A memoryless strategy for choosing actions
	"""
	TMDPOperations = TypeVar("TMDPOperations",bound=MDPOperationsInterface[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction])
	TMDPActionStrategy = TypeVar("TMDPActionStrategy",bound="MDPActionStrategyInterface[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction]")
	# Abstract class
	def __init__(self) -> None:
		pass
	def deepCopy(self: TMDPActionStrategy) -> TMDPActionStrategy:
		return self
	def getMDPAction(self, mdpState: TMDPState, mdpOperations: TMDPOperations, quietInfoStr: bool) -> Optional[TMDPAction]:
		"""!
		The strategy will receive an MDPOperations instance and
		must return a legal MDPAction
		"""
		mdpActions=mdpOperations.getLegalActions(mdpState)
		return self._getMDPActionInSubset(mdpActions, mdpState, mdpOperations, quietInfoStr)
	def getMDPActionInSubset(self, mdpActions: List[TMDPAction], mdpState: TMDPState, mdpOperations: TMDPOperations, quietInfoStr: bool) -> Optional[TMDPAction]:
		"""!
		The strategy will receive a set of actions and an MDPOperations instance and
		must return a legal MDPAction from the set
		"""
		legal=mdpOperations.getLegalActions(mdpState)
		mdpLegalActions=[]
		for a in mdpActions:
			if a in legal:
				mdpLegalActions.append(a)
		return self._getMDPActionInSubset(mdpLegalActions, mdpState, mdpOperations, quietInfoStr)
	def _getMDPActionInSubset(self, mdpActions: List[TMDPAction], mdpState: TMDPState, mdpOperations: TMDPOperations, quietInfoStr: bool) -> Optional[TMDPAction]:
		"""!
		The Agent will receive an MDPOperations and
		must return an action taken from a set of legal mdpActions
		"""
		raiseNotDefined()
		return None


class MDPProbabilisticActionStrategyInterface( MDPActionStrategyInterface[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction] ):
	"""!
	A probabilistic strategy that draws from a distribution over legal actions
	"""
	TMDPOperations = TypeVar("TMDPOperations",bound=MDPOperationsInterface[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction])
	# Abstract class
	def _getMDPActionInSubset(self, mdpActions: List[TMDPAction], mdpState: TMDPState, mdpOperations: TMDPOperations, quietInfoStr: bool) -> Optional[TMDPAction]:
		dist = self._getDistribution(mdpActions, mdpState, mdpOperations)
		if len(dist) == 0:
			return None
			# raise Exception("empty distribution")
		choice = util.chooseFromDistribution( dist ).deepCopy()
		if not quietInfoStr:
			isUniversal = True
			vv = None
			for k,v in dist.items():
				if vv is None : vv = v
				if v != vv: isUniversal = False
			if isUniversal:
				if choice.infoStr != '':
					choice.infoStr += '#'
				choice.infoStr += 'DistUniversal'
			else:
				if choice.infoStr != '':
					choice.infoStr += '#'
				choice.infoStr += 'Dist'+str(dist)
		return choice
	def _getDistribution(self, mdpActions: List[TMDPAction], mdpState: TMDPState, mdpOperations: TMDPOperations) -> util.ConsoleStrFloatCounter[TMDPAction]:
		"Returns a Counter from util.py encoding a distribution over legal actions."
		raiseNotDefined()
		return util.ConsoleStrFloatCounter()


class MDPUniformActionStrategy( MDPProbabilisticActionStrategyInterface[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction] ):
	"""!
	A strategy that chooses a legal action uniformly at random.
	"""
	TMDPOperations = TypeVar("TMDPOperations",bound=MDPOperationsInterface[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction])
	def _getDistribution(self, mdpActions: List[TMDPAction], mdpState: TMDPState, mdpOperations: TMDPOperations) -> util.ConsoleStrFloatCounter[TMDPAction]:
		dist: util.ConsoleStrFloatCounter[TMDPAction] = util.ConsoleStrFloatCounter()
		for a in mdpActions: dist[a] = 1.0
		dist.normalize()
		return dist

class MDPActionAdviceInterface(Generic[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction]):
	"""!
	An advice (nondeterministic strategy) for choosing actions
	"""
	TMDPOperations = TypeVar("TMDPOperations",bound=MDPOperationsInterface[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction])
	TMDPActionAdvice = TypeVar("TMDPActionAdvice",bound="MDPActionAdviceInterface[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction]")
	def __init__(self) -> None:
		pass
	def deepCopy(self: TMDPActionAdvice) -> TMDPActionAdvice:
		return self
	def getMDPActionAdvice(self, mdpState: TMDPState, mdpOperations: TMDPOperations, quietInfoStr: bool) -> Tuple[List[TMDPAction],List[TMDPAction]]:
		"""!
		The strategy will receive an MDPOperations instance and
		must return a set of legal MDPActions
		"""
		mdpActions=mdpOperations.getLegalActions(mdpState)
		choices = self._getMDPActionAdviceInSubset(mdpActions, mdpState, mdpOperations)
		if not quietInfoStr:
			if len(choices)==len(mdpActions):
				for mdpAction in choices:
					if mdpAction.infoStr != '':
						mdpAction.infoStr += '#'
					mdpAction.infoStr += 'AdviceFull'
			else:
				for mdpAction in choices:
					if mdpAction.infoStr != '':
						mdpAction.infoStr += '#'
					mdpAction.infoStr += 'Advice['+','.join([mdpAction.consoleStr() for mdpAction in choices])+']'
		return choices, mdpActions
	def getMDPActionAdviceInSubset(self, mdpActions: List[TMDPAction], mdpState: TMDPState, mdpOperations: TMDPOperations, quietInfoStr: bool) -> Tuple[List[TMDPAction],List[TMDPAction]]:
		"""!
		The strategy will receive an MDPOperations instance and
		must return a set of legal MDPActions from a subset
		"""
		choices = self._getMDPActionAdviceInSubset(mdpActions, mdpState, mdpOperations)
		if not quietInfoStr:
			if len(choices)==len(mdpActions):
				for mdpAction in choices:
					if mdpAction.infoStr != '':
						mdpAction.infoStr += '#'
					mdpAction.infoStr += 'AdviceFull'
			else:
				for mdpAction in choices:
					if mdpAction.infoStr != '':
						mdpAction.infoStr += '#'
					mdpAction.infoStr += 'Advice['+','.join([mdpAction.consoleStr() for mdpAction in choices])+']'
		return choices, mdpActions
	def _getMDPActionAdviceInSubset(self, mdpActions: List[TMDPAction], mdpState: TMDPState, mdpOperations: TMDPOperations) -> List[TMDPAction]:
		"""!
		The Agent will receive an MDPOperations and
		must return a subset of a set of legal mdpActions
		"""
		choices = []
		for mdpAction in mdpActions:
			if self._isMDPActionAllowed(mdpAction, mdpState, mdpOperations):
				choices.append(mdpAction.deepCopy())
		return choices
	def _isMDPActionAllowed(self, mdpAction: TMDPAction, mdpState: TMDPState, mdpOperations: TMDPOperations) -> bool:
		# raiseNotDefined()
		return True


class MDPFullActionAdvice( MDPActionAdviceInterface[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction] ):
	"""!
	A trivial advice that allow everything
	"""
	TMDPOperations = TypeVar("TMDPOperations",bound=MDPOperationsInterface[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction])
	def getMDPActionAdvice(self, mdpState: TMDPState, mdpOperations: TMDPOperations, quietInfoStr: bool) -> List[TMDPAction]:
		"""
		The strategy will receive an MDPOperations instance and
		must return a set of legal MDPActions
		"""
		choices = mdpOperations.getLegalActions(mdpState)
		if not quietInfoStr:
			for mdpAction in choices:
				if mdpAction.infoStr != '':
					mdpAction.infoStr += '#'
				mdpAction.infoStr += 'AdviceFull'
		return choices, choices


class MDPPathAdviceInterface(Generic[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction]):
	"""!
	An advice that allows or rejects paths
	"""
	TMDPPathAdvice = TypeVar("TMDPPathAdvice",bound="MDPPathAdviceInterface[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction]")
	def __init__(self) -> None:
		pass
	def deepCopy(self: TMDPPathAdvice) -> TMDPPathAdvice:
		return self
	def isValidPath(self, mdpExecutionEngine: MDPExecutionEngine[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction]) -> bool:
		"""
		The strategy will receive an MDPExecutionEngine instance
		"""
		raiseNotDefined()
		return True


class MDPFullPathAdvice(MDPPathAdviceInterface[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction]):
	def isValidPath(self, mdpExecutionEngine: MDPExecutionEngine[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction]) -> bool:
		return True

class MDPEmptyPathAdvice(MDPPathAdviceInterface[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction]):
	def isValidPath(self, mdpExecutionEngine: MDPExecutionEngine[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction]) -> bool:
		return False

class MDPNonTerminalPathAdvice(MDPPathAdviceInterface[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction]):
	def isValidPath(self, mdpExecutionEngine: MDPExecutionEngine[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction]) -> bool:
		return not mdpExecutionEngine.isTerminal()

class MDPAboveThresPathAdvice(MDPPathAdviceInterface[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction]):
	def __init__(self, threshold:float) -> None:
		self.threshold = threshold
	def deepCopy(self) -> "MDPAboveThresPathAdvice[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction]":
		return MDPAboveThresPathAdvice(self.threshold)

	def isValidPath(self, mdpExecutionEngine: MDPExecutionEngine[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction]) -> bool:
		return (mdpExecutionEngine.mdpPathReward() >= self.threshold)


class MDPStateScoreInterface(Generic[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction]):
	"""!
	A state score for a given agent
	"""
	TMDPStateScore = TypeVar("TMDPStateScore",bound="MDPStateScoreInterface[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction]")
	def __init__(self) -> None:
		pass
	def deepCopy(self: TMDPStateScore) -> TMDPStateScore:
		return self

	def getScore( self, executionEngine: MDPExecutionEngine[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction] ) -> float:
		"""
		The Agent will receive an executionEngine and
		must return a score
		"""
		raiseNotDefined()
		endState = executionEngine.mdpEndState()
		mdpHorizonReward=0.0
		return mdpHorizonReward
		# return executionEngine.getHorizonReward()

class MDPStateScoreZero(MDPStateScoreInterface[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction]):
	def getScore(self, executionEngine: MDPExecutionEngine[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction]) -> float:
		return 0.0

class OptionsSimulationEngine(Generic[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction]):
	TMDPActionStrategy = TypeVar("TMDPActionStrategy",bound=MDPActionStrategyInterface[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction])
	TMDPActionAdvice = TypeVar("TMDPActionAdvice",bound=MDPActionAdviceInterface[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction])
	TMDPPathAdvice = TypeVar("TMDPPathAdvice",bound=MDPPathAdviceInterface[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction])
	TMDPStateScore = TypeVar("TMDPStateScore",bound=MDPStateScoreInterface[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction])
	def __init__(self, horizon: int, ignoreNonDecisionStates: bool, mdpActionStrategy: TMDPActionStrategy, mdpActionAdvice: TMDPActionAdvice, mdpPathAdvice: TMDPPathAdvice, mdpStateScore: TMDPStateScore, alpha: float, rejectFactor: int, quiet: bool, quietInfoStr: bool, printEachStep: bool, printCompact: bool) -> None:
		self.horizon = horizon # horizon for the simulations
		self.ignoreNonDecisionStates = ignoreNonDecisionStates # if true, we only count states with multiple available actions for horizon purposes
		self.mdpActionStrategy = mdpActionStrategy # an action strategy
		self.mdpActionAdvice = mdpActionAdvice # an advice on actions (prunes actions)
		self.mdpPathAdvice = mdpPathAdvice # an advice on paths (prunes paths)
		self.rejectFactor = rejectFactor # multiplies the number of simulations to get the amount of tries under a path advice
		self.mdpStateScore = mdpStateScore # a score to add for an execution during simulation
		self.alpha = alpha # coefficient of stateScore, 0 = no stateScore
		self.quiet = quiet
		self.quietInfoStr = quietInfoStr
		self.printEachStep = printEachStep
		self.printCompact = printCompact
	def deepCopy(self) -> "OptionsSimulationEngine[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction]":
		return OptionsSimulationEngine(horizon = self.horizon, ignoreNonDecisionStates = self.ignoreNonDecisionStates, mdpActionStrategy = self.mdpActionStrategy.deepCopy(), mdpActionAdvice = self.mdpActionAdvice.deepCopy(), mdpPathAdvice = self.mdpPathAdvice.deepCopy(), rejectFactor = self.rejectFactor, mdpStateScore = self.mdpStateScore.deepCopy(), alpha = self.alpha, quiet = self.quiet, quietInfoStr = self.quietInfoStr, printEachStep=self.printEachStep, printCompact=self.printCompact)

class MDPSimulationEngine(Generic[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction]):
	"""!
	MDPSimulationEngine is used to draw simulations that extend an execution,
	according to an action strategy.
	The execution is sequentially extended and reset after each simulation,
	Use execution.executionCopy() if running several MDPSimulationEngine instances in parallel
	to give independent instances to the constructors.
	"""
	def __init__(self, mdpExecutionEngine: MDPExecutionEngine[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction], options: OptionsSimulationEngine[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction]) -> None:
		self.mdpExecutionEngine = mdpExecutionEngine # an ExecutionEngine instance used to play the new decisions
		self.horizon = options.horizon # horizon for the simulations
		self.ignoreNonDecisionStates = options.ignoreNonDecisionStates # if true, we only count states with multiple available actions for horizon purposes
		self.mdpActionStrategy = options.mdpActionStrategy # an action strategy
		self.mdpActionAdvice = options.mdpActionAdvice # an advice on actions (prunes actions)
		self.mdpPathAdvice = options.mdpPathAdvice # an advice on paths (prunes paths)
		self.rejectFactor = options.rejectFactor # multiplies the number of simulations to get the amount of tries under a path advice
		self.mdpStateScore = options.mdpStateScore # a score to add for an execution during simulation
		self.alpha = options.alpha # coefficient of stateScore, 0 = no stateScore
		self.quiet = options.quiet
		self.quietInfoStr = options.quietInfoStr
		self.printEachStep = options.printEachStep
		self.printCompact = options.printCompact

	def _getMDPAction(self) -> TMDPAction:
		nonDecisionAction = False
		mdpActions, mdpActionsFull = self.mdpActionAdvice.getMDPActionAdvice(self.mdpExecutionEngine.mdpEndState(), self.mdpExecutionEngine.mdpOperations,self.quietInfoStr)
		if self.ignoreNonDecisionStates and len(mdpActionsFull) <= 1:
			nonDecisionAction = True
		mdpAction = self.mdpActionStrategy.getMDPActionInSubset(mdpActions,self.mdpExecutionEngine.mdpEndState(), self.mdpExecutionEngine.mdpOperations,self.quietInfoStr)
		if mdpAction is None:
			if not self.quiet: print("action is none, trying again without action advice")
			mdpAction = self.mdpActionStrategy.getMDPAction(self.mdpExecutionEngine.mdpEndState(), self.mdpExecutionEngine.mdpOperations,self.quietInfoStr)
			if mdpAction is None:
				raise NoMoveException("could not get an action")
		return mdpAction, nonDecisionAction

	def _extend(self,abort: bool) -> int:
		"""!
		Simulates one time step: draw an action and play the new time step
		in the execution engine.
		"""
		mdpAction, nonDecisionAction = self._getMDPAction()
		if abort and not nonDecisionAction:
			if not self.quiet and self.printEachStep: print("_extend aborts and returns -1")
			return -1
		mdpTransition = self.mdpExecutionEngine.drawTransition(mdpAction, self.quietInfoStr)
		self.mdpExecutionEngine.append(mdpTransition,nonDecisionAction)
		if nonDecisionAction:
			if not self.quiet and self.printEachStep: print("_extend returns 0")
			return 0
		else:
			if not self.quiet and self.printEachStep: print("_extend returns 1")
			return 1

	def _runSimulation(self, timeI: int, timeIReal: int) -> None:
		"""!
		Simulates one full run until a given depth.
		"""
		# if not self.quiet: print("--------------------------------------------------------------")
		# if not self.quiet: print("simulating for",depth-timeI,"steps from state",self.mdpExecutionEngine.stateConsoleStr())

		depth = timeI
		j = timeIReal
		while depth <= self.horizon:
			if not self.quiet and self.printEachStep: print("depth",depth,"j",j,"horizon",self.horizon)
			if self.mdpExecutionEngine.isTerminal():
				break
			v = self._extend((depth == self.horizon))
			if v < 0:
				break
			depth += v
			if not self.quiet and self.printEachStep :
				 print("step",j-timeIReal,"(",depth,"/",self.horizon,"):",self.mdpExecutionEngine.lastConsoleStr())
			j += 1
		# if not self.quiet: print("--------------------------------------------------------------")

	def getSimulations(self, numSims: int) -> List[Tuple[MDPExecutionEngine[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction], float, int]]:
		"""!
		Draws numSims simulations.
		Returns a list of pairs (mdpExecutionEngineS,numS),
		so that mdpExecutionEngineS contains a simulation until horizon depth,
		and so that the sum of numS equals numSims.
		"""
		if not self.quiet: print("[===========================================================[")
		timeI=self.mdpExecutionEngine.length(ignoreNonDecisionStates = self.ignoreNonDecisionStates)
		timeIReal=self.mdpExecutionEngine.length(ignoreNonDecisionStates = False)
		fastResetDataI=self.mdpExecutionEngine.getFastResetData()
		results: List[Tuple[MDPExecutionEngine[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction], float, int]]=[]
		# if not self.quiet: print("timeI",timeI,"timeIReal",timeIReal,"horizon",self.horizon)
		if not self.mdpExecutionEngine.isTerminal() and timeI<=self.horizon:
			if not self.quiet: print("running",numSims,"simulations for depth",timeI,"to",self.horizon,"from",self.mdpExecutionEngine.stateConsoleStr())
			numResults = 0
			for s in range(numSims * self.rejectFactor):
				if s>0:
					self.mdpExecutionEngine.fastReset(fastResetDataI)
				# if not self.quiet: print("===========================================================")
				# if not self.quiet: print("running simulation for depth",self.horizon,"from",self.mdpExecutionEngine.stateConsoleStr())
				self._runSimulation(timeI,timeIReal)

				if self.mdpPathAdvice.isValidPath(self.mdpExecutionEngine):
					mdpExecutionEngine = self.mdpExecutionEngine.executionCopy()
					mdpPathReward=mdpExecutionEngine.mdpPathReward()
					stateScore=self.mdpStateScore.getScore(mdpExecutionEngine)
					mdpReward = (1-self.alpha)*mdpPathReward + self.alpha*stateScore
					results.append((mdpExecutionEngine,mdpReward,1))
					numResults += 1
					if not self.quiet: #and not self.printEachStep :
						if not self.printCompact:
							print("|\t",self.mdpExecutionEngine.suffixConsoleStr(timeIReal),"stateScore","{:.2f}".format(stateScore),"weight",self.alpha,"->","{:.2f}".format(mdpReward))
						else:
							print("|\t",self.mdpExecutionEngine.length(ignoreNonDecisionStates = False),"steps to",self.mdpExecutionEngine.stateConsoleStr(),"stateScore","{:.2f}".format(stateScore),"weight",self.alpha,"->","{:.2f}".format(mdpReward))

				if numResults >= numSims:
					break
			if not self.quiet: print("ran",s+1,"simulations")
		else:
			mdpExecutionEngine = self.mdpExecutionEngine.executionCopy()
			mdpPathReward=mdpExecutionEngine.mdpPathReward()
			stateScore=self.mdpStateScore.getScore(mdpExecutionEngine)
			mdpReward = (1-self.alpha)*mdpPathReward + self.alpha*stateScore
			results=[(mdpExecutionEngine,mdpReward,numSims)]
			if not self.quiet: #and not self.printEachStep:
				print("|\t",self.mdpExecutionEngine.suffixConsoleStr(timeIReal),"stateScore","{:.2f}".format(stateScore),"weight",self.alpha,"->","{:.2f}".format(mdpReward),"|*",numSims)
		self.mdpExecutionEngine.fastReset(fastResetDataI)
		if not self.quiet: print("]===========================================================]")
		return results
	def getSimulationReward(self, numSims: int) -> Optional[float]:
		"""!
		Draws numSims simulations until horizon depth
		and returns the average mdpReward obtained.
		"""
		if numSims<=0:
			raise Exception("zero simulations is not enough")
			# return 0
		mdpRewardEstimates = 0.0
		numSelect = 0
		# timeI=self.mdpExecutionEngine.length(ignoreNonDecisionStates = self.ignoreNonDecisionStates)
		simulations=self.getSimulations(numSims)
		# if not self.quiet: print("obtained",len(simulations),"simulations:")
		for mdpExecutionEngineS,rewardS,numS in simulations:
			# if not self.quiet: print("|\t",mdpExecutionEngineS.suffixConsoleStr(timeI),"*",numS)
			mdpRewardEstimates+=rewardS*numS
			numSelect+=numS
		if numSelect!=numSims:
			if not self.quiet: print("found",numSelect,"valid simulations out of",numSims)
			return None
			# raise Exception("invalid number of simulations obtained")
		mdpRewardEstimates/= numSelect
		return mdpRewardEstimates


class MDPMCTSNode(Generic[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction]):
	"""!
	a node in the mcts tree
	"""
	def __init__(self, mdpState: TMDPState, mdpParentNode: "Optional[MDPMCTSNode[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction]]", mdpParentTransition: Optional[MDPTransition[TMDPAction, TMDPStochasticAction]], depth: int, legalActions: List[TMDPAction], nonDecisionAction : bool) -> None:
		self.mdpState = mdpState # a MDPState instance
		self.mdpParentNode = mdpParentNode # an MDPMCTSNode
		self.mdpParentTransition = mdpParentTransition # a MDPTransition that leads from mdpParentNode to self
		self.depth = depth # depth of current node
		self.legalActions = legalActions # list of available actions from this node
		self.numVisits = 0 # MCTS counter
		self.totalScore = 0.0 # MCTS counter
		self.children: "Dict[TMDPAction, List[MDPMCTSNode[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction]]]"  = {} # maps an action to a list of children MDPMCTSNodes
		self.actionVisits: Dict[TMDPAction, int] = {}
		self.actionScore: Dict[TMDPAction, float] = {}
		self.nonDecisionAction = nonDecisionAction

	@classmethod
	def rootFromExec(cls, mdpExecutionEngine: MDPExecutionEngine[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction], legalActions: List[TMDPAction], nonDecisionAction : bool) -> "MDPMCTSNode[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction]":
		if mdpExecutionEngine.length(ignoreNonDecisionStates = False)>0:
			raise Exception("bad MCTS root")
		mdpState = mdpExecutionEngine.mdpEndState().deepCopy()
		mdpParentNode = None
		mdpParentTransition = None
		depth = 0
		return cls(mdpState, mdpParentNode, mdpParentTransition, depth, legalActions, nonDecisionAction)

	def __str__(self) -> str:
		strSelf = '\nNode: '+str(hash(self))
		strState = '\nState: '+str(self.mdpState)
		if not self.mdpParentNode is None:
			strParent = '\nParent node: ' + str(hash(self.mdpParentNode))
		else:
			strParent = '\nParent node: ' + str(None)
		strParentAction = '\nParent transition: ' + str(self.mdpParentTransition)
		strDepth = '\nDepth: '+str(self.depth)
		strLegalActions = '\nLegal actions: ' + ' '.join([str(a) for a in self.legalActions])
		strActionVisits = '\nAction Visits: {' + ' '.join([str(k)+'='+str(v) for k,v in self.actionVisits.items()]) + '}'
		strActionScore = '\nAction Score: {' + ' '.join([str(k)+'='+str(v) for k,v in self.actionScore.items()]) + '}'
		strPrint = strSelf + strState + strParent + strParentAction + strDepth + strLegalActions + '\nnumVisits: '+str(self.numVisits) + '\ntotalScore: '+str(self.totalScore) + strActionVisits + strActionScore
		return (strPrint + '\n')

	def consoleStr(self, bUpwards: bool = False, bDownwards: bool = False) -> str:
		if bUpwards and bDownwards:
			raise Exception("cannot recursively go both ways")
		if bUpwards or bDownwards:
			tab = '\t'*self.depth
		else:
			tab = ''
		strSelf = '\n'+tab+'Node: '+str(hash(self))
		strState = '\n'+tab+'State: '+self.mdpState.consoleStr()
		if not self.mdpParentNode is None:
			if bUpwards:
				strParent = '\n'+tab+'Parent node:' + self.mdpParentNode.consoleStr(bUpwards,bDownwards)
			else:
				strParent = '\n'+tab+'Parent node: ' + str(hash(self.mdpParentNode))
			if self.mdpParentTransition is None:
				raise Exception("not at root and no parent transition")
			strParentAction = '\n'+tab+'Parent transition: ' + self.mdpParentTransition.consoleStr()
		else:
			strParent = '\n'+tab+'Parent node: ' + str(None)
			strParentAction = '\n'+tab+'Parent transition: ' + str(None)
		strDepth = '\n'+tab+'Depth: '+str(self.depth)
		strLegalActions = '\n'+tab+'Legal actions: ' + ' '.join([a.consoleStr() for a in self.legalActions])
		strActionVisits = '\n'+tab+'Action Visits: {' + ' '.join([k.consoleStr()+'='+str(v) for k,v in self.actionVisits.items()]) + '}'
		strActionScore = '\n'+tab+'Action Score: {' + ' '.join([k.consoleStr()+'='+str(v) for k,v in self.actionScore.items()]) + '}'
		if len(self.children.items()) > 0:
			if bDownwards:
				strChildrens = '\n'+tab+'Childrens: {\n'+tab + ('\n'+tab).join([k.consoleStr()+'=\n'+tab+''.join([n.consoleStr(bUpwards,bDownwards) for n in v]) for k,v in self.children.items()]) + '\n'+tab+'}\n'
			else:
				strChildrens = '\n'+tab+'Childrens: {' + (';').join([k.consoleStr()+':'+' '.join([str(hash(n)) for n in v]) for k,v in self.children.items()]) + '}\n'
		else:
			strChildrens = '\n'+tab+'Childrens: {}\n'
		strPrint = strParent + strSelf + strState + strParentAction + strDepth + strLegalActions + '\n'+tab+'numVisits: '+str(self.numVisits) + '\n'+tab+'totalScore: '+str(self.totalScore) + strActionVisits + strActionScore + strChildrens
		return ('\n'+tab+'================'+strPrint +tab+'================\n')


class OptionsMCTSEngine(Generic[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction]):
	TMDPActionStrategy = TypeVar("TMDPActionStrategy",bound=MDPActionStrategyInterface[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction])
	TMDPActionAdvice = TypeVar("TMDPActionAdvice",bound=MDPActionAdviceInterface[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction])
	TMDPPathAdvice = TypeVar("TMDPPathAdvice",bound=MDPPathAdviceInterface[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction])
	TMDPStateScore = TypeVar("TMDPStateScore",bound=MDPStateScoreInterface[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction])
	def __init__(self, horizon: int, ignoreNonDecisionStates : bool, mctsConstant: float, mdpActionStrategy: TMDPActionStrategy, mdpActionAdvice: TMDPActionAdvice, mdpActionAdviceRoot: TMDPActionAdvice, optionsSimulationEngine: OptionsSimulationEngine[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction], mdpThresholdActionAdvice: Optional[TMDPActionAdvice], mdpThresholdPathAdvice: Optional[TMDPPathAdvice], quiet: bool, quietInfoStr: bool) -> None:
		self.horizon = horizon
		self.ignoreNonDecisionStates = ignoreNonDecisionStates # if true, we only count states with multiple available actions for horizon purposes
		self.mctsConstant = mctsConstant # the constant used by UCB
		self.mdpActionStrategy = mdpActionStrategy # an action strategy
		self.mdpActionAdvice = mdpActionAdvice # an advice on actions (prunes actions)
		self.mdpActionAdviceRoot = mdpActionAdviceRoot # an advice on actions at root node (prunes actions)
		self.optionsSimulationEngine = optionsSimulationEngine
		self.mdpThresholdActionAdvice = mdpThresholdActionAdvice # action advice to create a threshold: optional
		self.mdpThresholdPathAdvice = mdpThresholdPathAdvice # path advice to create a threshold: optional
		self.quiet = quiet
		self.quietInfoStr = quietInfoStr
	def deepCopy(self) -> "OptionsMCTSEngine[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction]":
		mdpThresholdActionAdvice = (self.mdpThresholdActionAdvice.deepCopy() if not self.mdpThresholdActionAdvice is None else None)
		mdpThresholdPathAdvice = (self.mdpThresholdPathAdvice.deepCopy() if not self.mdpThresholdPathAdvice is None else None)
		return OptionsMCTSEngine(horizon=self.horizon, ignoreNonDecisionStates=self.ignoreNonDecisionStates, mctsConstant=self.mctsConstant, mdpActionStrategy=self.mdpActionStrategy, mdpActionAdvice=self.mdpActionAdvice.deepCopy(),mdpActionAdviceRoot=self.mdpActionAdviceRoot.deepCopy(), optionsSimulationEngine=self.optionsSimulationEngine.deepCopy(), mdpThresholdActionAdvice=mdpThresholdActionAdvice, mdpThresholdPathAdvice=mdpThresholdPathAdvice, quiet=self.quiet, quietInfoStr=self.quietInfoStr)

class MCTSEngine(Generic[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction]):
	"""
	MCTSEngine
	"""
	# TMDPActionStrategy = TypeVar("TMDPActionStrategy",bound=MDPActionStrategyInterface[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction])
	# TMDPActionAdvice = TypeVar("TMDPActionAdvice",bound=MDPActionAdviceInterface[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction])
	TMDPPathAdvice = TypeVar("TMDPPathAdvice",bound=MDPPathAdviceInterface[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction])
	# TMDPStateScore = TypeVar("TMDPStateScore",bound=MDPStateScoreInterface[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction])
	def __init__(self, mdpExecutionEngine: MDPExecutionEngine[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction], options: OptionsMCTSEngine[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction]) -> None:
		self.mdpExecutionEngine = mdpExecutionEngine # an ExecutionEngine instance used to construct the tree
		mdpActions,mdpActionsFull = options.mdpActionAdviceRoot.getMDPActionAdvice(self.mdpExecutionEngine.mdpEndState(), self.mdpExecutionEngine.mdpOperations, options.quietInfoStr)
		mdpActions,rootActionsFull = options.mdpActionAdvice.getMDPActionAdviceInSubset(mdpActions, self.mdpExecutionEngine.mdpEndState(), self.mdpExecutionEngine.mdpOperations, options.quietInfoStr)

		nonDecisionAction = (len(mdpActionsFull)<=1) and options.ignoreNonDecisionStates
		self.root = MDPMCTSNode.rootFromExec(self.mdpExecutionEngine,mdpActions,nonDecisionAction) # an MDPMCTSNode instance for the root of MCTS

		self.horizon = options.horizon
		self.ignoreNonDecisionStates = options.ignoreNonDecisionStates # if true, we only count states with multiple available actions for horizon purposes
		self.mctsConstant = options.mctsConstant # the constant used by UCB
		self.mdpActionStrategySelection = options.mdpActionStrategy # an action strategy
		self.mdpActionAdviceSelection = options.mdpActionAdvice # an advice on actions (prunes actions)
		self.optionsSimulationEngine = options.optionsSimulationEngine
		self.mdpThresholdActionAdvice = options.mdpThresholdActionAdvice # action advice to create a threshold: optional
		self.mdpThresholdPathAdvice = options.mdpThresholdPathAdvice # path advice to create a threshold: optional
		self.quiet = options.quiet
		self.quietInfoStr = options.quietInfoStr

	def _getMDPActionSelection(self, mdpActions: List[TMDPAction]) -> TMDPAction:
		mdpAction = self.mdpActionStrategySelection.getMDPActionInSubset(mdpActions,self.mdpExecutionEngine.mdpEndState(), self.mdpExecutionEngine.mdpOperations, self.quietInfoStr)
		if mdpAction is None:
			if not self.quiet: print("action is none, trying again without action advice")
			mdpAction = self.mdpActionStrategySelection.getMDPAction(self.mdpExecutionEngine.mdpEndState(), self.mdpExecutionEngine.mdpOperations, self.quietInfoStr)
			if mdpAction is None:
				raise Exception("could not get an action")
		return mdpAction

	def doMCTSIteration(self, numSims: int) -> None:
		node=self.root
		executionEngine=self.mdpExecutionEngine
		fastResetData0=executionEngine.getFastResetData()
		# Selection phase
		# if not self.quiet: print("MCTS tree before selection:",self.root.consoleStr(bDownwards=True))
		if not self.quiet: print("Selection phase")
		depth = 0
		decisionDepth = 0
		while decisionDepth < self.horizon:
			# Selects an action
			legal = node.legalActions
			depth += 1
			if len(legal) > 1 or not self.ignoreNonDecisionStates:
				decisionDepth += 1
			nonUcbScores: util.ConsoleStrFloatCounter[TMDPAction] = util.ConsoleStrFloatCounter()
			ucbScores: util.ConsoleStrFloatCounter[TMDPAction] = util.ConsoleStrFloatCounter()
			notExplored: List[TMDPAction] = []
			for actionLegal in legal:
				action = actionLegal.deepCopy()
				# if not self.quietInfoStr:
				if action in node.actionVisits and node.actionVisits[action]>0:
					nonUcbScores[action] = node.actionScore[action] / node.actionVisits[action]
					ucbScores[action] = node.actionScore[action] / node.actionVisits[action] + self.mctsConstant * math.sqrt(2 * math.log(node.numVisits) / node.actionVisits[action])
				else:
					notExplored.append(action)
					nonUcbScores[action] = float('inf')
					ucbScores[action] = float('inf')
			if len(notExplored)>0:
				actionChosen = self._getMDPActionSelection(notExplored)
				if not self.quietInfoStr:
					if actionChosen.infoStr != '':
						actionChosen.infoStr += '#'
					actionChosen.infoStr += 'Scores'+str(nonUcbScores)+'UCB'+str(ucbScores)+'NEW' #'NEW['+','.join([a.miniConsoleStr() for a in notExplored])+']'
			else:
				argMax=ucbScores.argMax()
				if len(argMax) == 0:
					raise NoMoveException("no move in argMax "+str(node))
				if len(argMax)>1:
					actionChosen = self._getMDPActionSelection(argMax)
					if not self.quietInfoStr:
						if actionChosen.infoStr != '':
							actionChosen.infoStr += '#'
						actionChosen.infoStr += 'Scores'+str(nonUcbScores)+'UCB'+str(ucbScores)+'TIE' #'['+','.join([a.miniConsoleStr() for a in argMax])+']'
				else:
					actionChosen=argMax[0]
					if not self.quietInfoStr:
						if actionChosen.infoStr != '':
							actionChosen.infoStr += '#'
						actionChosen.infoStr += 'Scores'+str(nonUcbScores)+'UCB'+str(ucbScores)
			# Draw a child of actionChosen
			mdpTransition = self.mdpExecutionEngine.drawTransition(actionChosen, self.quietInfoStr)
			if not self.quiet: print("+\t",mdpTransition.consoleStr())

			# go to next node in execution engine
			executionEngine.append(mdpTransition,node.nonDecisionAction)
			# Search for next node in children
			nextNode=None
			newAction=True
			if actionChosen in node.children:
				newAction=False
				for child in node.children[actionChosen]:
					if child.mdpParentTransition == mdpTransition:
						nextNode=child
			if not nextNode is None:
				if not self.quiet: print("->\tfound child in MCTS tree")
				node=nextNode
			else:
				# Constructs the new node
				legalActions,legalActionsFull = self.mdpActionAdviceSelection.getMDPActionAdvice(executionEngine.mdpEndState(), executionEngine.mdpOperations, self.quietInfoStr)

				nonDecisionAction = (len(legalActionsFull)<=1) and self.ignoreNonDecisionStates
				mdpParentTransition = mdpTransition.deepCopy()
				if not self.quietInfoStr:
					mdpParentTransition.mdpAction.infoStr = "" # resets the extra info
					mdpParentTransition.mdpStochasticAction.infoStr = "" # resets the extra info
				newNode = MDPMCTSNode(executionEngine.mdpEndState().deepCopy(),node,mdpParentTransition,depth,legalActions,nonDecisionAction)
				if newAction:
					actionCopy = actionChosen.deepCopy()
					if not self.quietInfoStr:
					 	actionCopy.infoStr = '' # erase extra info
					node.actionVisits[actionCopy]=0
					node.actionScore[actionCopy]=0
					node.children[actionCopy]=[newNode]
				else:
					node.children[actionChosen].append(newNode)
				if not self.quiet: print("->\tnew child added to MCTS tree")
				node=newNode
				if not nonDecisionAction:
					break
			if executionEngine.isTerminal():
				break
		# if not self.quiet: print("MCTS selected path in tree:",node.consoleStr(bUpwards=True))
		# Simulation phase
		# if not self.quiet: print("MCTS tree before simulation:",self.root.consoleStr(bDownwards=True))
		if not self.quiet: print("Simulation phase")
		options = self.optionsSimulationEngine.deepCopy()
		if (self.mdpThresholdActionAdvice is None) != (self.mdpThresholdPathAdvice is None):
			raise Exception('bad threshold advices, define both or neither')
		if not self.mdpThresholdActionAdvice is None:
			if self.mdpThresholdPathAdvice is None: # to make the type-checker happy
				raise Exception('bad threshold advices, define both or neither')
			optionsT = options.deepCopy()
			mdpActionAdvice = self.mdpThresholdActionAdvice
			optionsT.mdpActionAdvice = mdpActionAdvice
			optionsT.mdpPathAdvice = self.mdpThresholdPathAdvice
			simEngineT = MDPSimulationEngine(executionEngine, optionsT) # engine to get reward threshold
			simulationRewardT=simEngineT.getSimulationReward(numSims)
			if simulationRewardT is None:
				if not self.quiet: print("Simulation for threshold output None, trying again with full path advice")
				tmp = MDPFullPathAdvice() # type: Any
				# tmp = MDPFullPathAdvice()  # type: MDPFullPathAdvice[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction]
				optionsT.mdpPathAdvice = tmp
				simEngineT = MDPSimulationEngine(executionEngine, optionsT)
				simulationRewardT=simEngineT.getSimulationReward(numSims)
				if simulationRewardT is None:
					raise Exception("could not get a reward estimate from simulation")
			tmp2 = MDPAboveThresPathAdvice(simulationRewardT) # type: Any
			options.mdpPathAdvice = tmp2
		simEngine = MDPSimulationEngine(executionEngine, options)

		simulationReward=simEngine.getSimulationReward(numSims)
		if simulationReward is None:
			if not self.quiet: print("Simulation output None, trying again with full path advice")
			tmp3 = MDPFullPathAdvice() # type: Any
			options.mdpPathAdvice = tmp3
			simEngine = MDPSimulationEngine(executionEngine, options)
			simulationReward=simEngine.getSimulationReward(numSims)
			if simulationReward is None:
				raise Exception("could not get a reward estimate from simulation")

		if not self.quiet: print("->\tsimulation outputs score",simulationReward)

		# Backpropagation phase
		# if not self.quiet: print("MCTS tree before backpropagation:",self.root.consoleStr(bDownwards=True))
		if not self.quiet: print("Backpropagation phase")
		node.numVisits += numSims
		node.totalScore += simulationReward * numSims
		while not node.mdpParentNode is None:
			if node.mdpParentTransition is None:
				raise Exception("bad tree shape")
			action=node.mdpParentTransition.mdpAction
			node.mdpParentNode.actionVisits[action] += numSims
			node.mdpParentNode.actionScore[action] += simulationReward * numSims
			node.mdpParentNode.numVisits += numSims
			node.mdpParentNode.totalScore += simulationReward * numSims
			node=node.mdpParentNode
		executionEngine.fastReset(fastResetData0)
		if not self.quiet: print("->\tnew root score for action",action.miniConsoleStr(),":",(node.actionScore[action] / node.actionVisits[action]))
		# if not self.quiet: print("MCTS tree after iteration:",self.root.consoleStr(bDownwards=True))


	def doMCTSIterations(self, numMCTSIters: int, numSims: int) -> None:
		for i in range(numMCTSIters):
			if not self.quiet:
				print("===========================================================")
				print("MCTS Iteration",i)
				print("===========================================================")
			self.doMCTSIteration(numSims)

	def getMCTSRootReward(self, action: TMDPAction) -> float:
		if action in self.root.actionVisits and self.root.actionVisits[action]>0:
			return self.root.actionScore[action] / self.root.actionVisits[action]
		return 0.0

	def getMCTSRootRewardDict(self):
		scoreEstimates: util.ConsoleStrFloatCounter[TMDPAction] = util.ConsoleStrFloatCounter()
		for action in  self.root.actionVisits:
			scoreEstimates[action] = self.getMCTSRootReward(action)
		return(scoreEstimates)

	def getMCTSRootAction(self) -> TMDPAction:
		scoreEstimates: util.ConsoleStrFloatCounter[TMDPAction] = util.ConsoleStrFloatCounter()
		for action in  self.root.actionVisits:
			scoreEstimates[action] = self.getMCTSRootReward(action)
		if len(scoreEstimates) == 0:
			raise Exception("no known actions at root of MCTS tree")
		argMax=scoreEstimates.argMax()
		if len(argMax) == 0:
			raise Exception("no move in argMax")
		actionChosen=random.choice(argMax)
		if not self.quietInfoStr:
			if actionChosen.infoStr != '':
				actionChosen.infoStr += '#'
			actionChosen.infoStr += 'Scores'+str(scoreEstimates)
		if not self.quiet:
			print("===========================================================")
			print("MCTS returns",actionChosen.consoleStr(),"from reward estimates",str(scoreEstimates))
			print("===========================================================")
		return actionChosen

class MDPMCTSActionStrategy( MDPActionStrategyInterface[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction] ):
	"""!
	A probabilistic strategy that draws from a distribution over legal actions
	"""
	TMDPOperations = TypeVar("TMDPOperations",bound=MDPOperationsInterface[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction])
	# Abstract class
	def __init__(self, numMCTSIters: int, numSims: int, optionsMCTSEngine: OptionsMCTSEngine[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction]) -> None:

		self.numMCTSIters=numMCTSIters
		self.numSims=numSims

		self.optionsMCTSEngine = optionsMCTSEngine
	def deepCopy(self) -> "MDPMCTSActionStrategy[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction]":
		return MDPMCTSActionStrategy(numMCTSIters=self.numMCTSIters, numSims=self.numSims, optionsMCTSEngine=self.optionsMCTSEngine)

	def getMDPValues(self, mdpState, mdpOperations, quietInfoStr):
		"""!
		The strategy will receive an MDPOperations instance and
		must return dict of values
		"""
		choices = mdpOperations.getLegalActions(mdpState)
		if len(choices) == 1:
			choice = choices[0]
			# if not self.optionsMCTSEngine.quiet:
			# 	print("===========================================================")
			# 	print('Only one available action; no need for MCTS:', choice.consoleStr())
			# 	print("===========================================================")
			return {choices[0]:1}
		predicates: List[TMDPPredicate] = []#[ProductMDPPredicate(p) for p in initialPredicateDatas]
		initState = mdpState.deepCopy()
		endState = mdpState.deepCopy()
		# TODO initial predicates
		mdpPath: MDPPath[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction] = MDPPath(initState,[],[predicates])
		execEngine = MDPExecutionEngine(mdpOperations,MDPExecution(mdpPath,endState,0,False,1),0)

		mctsEngine = MCTSEngine(execEngine, self.optionsMCTSEngine)
		mctsEngine.doMCTSIterations(self.numMCTSIters,self.numSims)
		return mctsEngine.getMCTSRootRewardDict()

	def getMDPAction(self, mdpState: TMDPState, mdpOperations: TMDPOperations, quietInfoStr: bool) -> TMDPAction:
		"""!
		The strategy will receive an MDPOperations instance and
		must return a legal MDPAction
		"""
		choices = mdpOperations.getLegalActions(mdpState)
		if len(choices) == 1:
			choice = choices[0]
			if not self.optionsMCTSEngine.quiet:
				print("===========================================================")
				print('Only one available action; no need for MCTS:', choice.consoleStr())
				print("===========================================================")
			return(choice)
		predicates: List[TMDPPredicate] = []#[ProductMDPPredicate(p) for p in initialPredicateDatas]
		initState = mdpState.deepCopy()
		endState = mdpState.deepCopy()
		# TODO initial predicates
		mdpPath: MDPPath[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction] = MDPPath(initState,[],[predicates])
		execEngine = MDPExecutionEngine(mdpOperations,MDPExecution(mdpPath,endState,0,False,1),0)

		mctsEngine = MCTSEngine(execEngine, self.optionsMCTSEngine)
		mctsEngine.doMCTSIterations(self.numMCTSIters,self.numSims)
		choice = mctsEngine.getMCTSRootAction()
		return choice
	def getMDPActionInSubset(self, mdpActions: List[TMDPAction], mdpState: TMDPState, mdpOperations: TMDPOperations, quietInfoStr: bool) -> TMDPAction:
		# raise Exception("MCTS strategy incompatible with advice")
		return self.getMDPAction(mdpState, mdpOperations, quietInfoStr)

class MDPMCTSTraceEngine(Generic[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction]):
	TMDPOperations = TypeVar("TMDPOperations",bound=MDPOperationsInterface[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction])

	def getMCTSSimulationEngine(self, mdpState: TMDPState, mdpPredicates: List[TMDPPredicate], mdpOperations: TMDPOperations, horizonTrace: int, numMCTSIters: int, numSims: int, optionsMCTSEngine: OptionsMCTSEngine[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction], quietTrace: bool, quietInfoStr: bool, printEachStepTrace: bool) -> MDPSimulationEngine[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction]:
		initState=mdpState.deepCopy()
		endState=mdpState.deepCopy()
		mdpPath: MDPPath[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction] = MDPPath(initState,[],[mdpPredicates])
		mdpExecutionEngine=MDPExecutionEngine(mdpOperations,MDPExecution(mdpPath,endState,0,False,1),0)
		mdpActionStrategy = MDPMCTSActionStrategy(numMCTSIters=numMCTSIters, numSims=numSims, optionsMCTSEngine=optionsMCTSEngine)
		mdpActionTraceAdvice=MDPFullActionAdvice() # type: MDPFullActionAdvice[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction]
		mdpPathTraceAdvice=MDPFullPathAdvice() # type: Any
		mdpStateScoreTrace=MDPStateScoreZero() # type: Any
		optionsTraceEngine: OptionsSimulationEngine[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction] = OptionsSimulationEngine(horizon=horizonTrace, ignoreNonDecisionStates = False, mdpActionStrategy=mdpActionStrategy, mdpActionAdvice=mdpActionTraceAdvice, mdpPathAdvice=mdpPathTraceAdvice, mdpStateScore=mdpStateScoreTrace, alpha = 0.0, rejectFactor=1, quiet=quietTrace, quietInfoStr=quietInfoStr, printEachStep=printEachStepTrace, printCompact=True)
		mdpSimulationEngine = MDPSimulationEngine(mdpExecutionEngine, optionsTraceEngine)
		return mdpSimulationEngine

	def runMCTSTrace(self, numTraces: int, mdpState: TMDPState, mdpPredicates: List[TMDPPredicate], mdpOperations: TMDPOperations, horizonTrace: int, numMCTSIters: int, numSims: int, optionsMCTSEngine: OptionsMCTSEngine[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction], quietTrace: bool, quietInfoStr: bool, printEachStepTrace: bool) -> List[Tuple[MDPExecutionEngine[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction], float, int]]:
		mdpSimulationEngine = self.getMCTSSimulationEngine(mdpState, mdpPredicates, mdpOperations, horizonTrace, numMCTSIters, numSims, optionsMCTSEngine,quietTrace,quietInfoStr,printEachStepTrace)
		results = mdpSimulationEngine.getSimulations(numTraces)
		return results
