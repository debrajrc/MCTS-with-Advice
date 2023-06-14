# mdpClasses.py

# from __future__ import annotations

import math
import random, sys

from typing import TypeVar, Type, Any, Optional, Sequence, List, Tuple, Dict, Union, Generic, NoReturn

import util
from util import raiseNotDefined

TMDPPredicate = TypeVar("TMDPPredicate",bound="MDPPredicateInterface")

##
# Abstract class
# Encodes predicates of the MDP
# Pacman: Win, Loss
class MDPPredicateInterface:

	# abstract methods that must be redefined
	def __init__(self) -> None:
		raiseNotDefined()
	def deepCopy(self: TMDPPredicate) -> TMDPPredicate:
		raiseNotDefined()
		return self
		# return MDPPredicateInterface()
	def initFromCopy(self: TMDPPredicate, other: TMDPPredicate) -> None:
		raiseNotDefined()
	def __str__(self) -> str:
		raiseNotDefined()
		return "()"

	# methods that can be redefined if needed
	def consoleStr(self) -> str:
		return str(self)
	def fastReset(self: TMDPPredicate, fastResetData: TMDPPredicate) -> None:
		self.initFromCopy(fastResetData)
	def getFastResetData(self: TMDPPredicate) -> TMDPPredicate:
		fastResetData = self.deepCopy()
		return fastResetData

	# define if you intend to write or read instances from a text file
	def fileStr(self) -> str:
		raiseNotDefined()
		return "p"
	@classmethod
	def fromFileStr(cls: Type[TMDPPredicate], s: str) -> TMDPPredicate:
		raiseNotDefined()
		if s=="":
			raise Exception("parsing error")
		return cls()


TMDPState = TypeVar("TMDPState",bound="MDPStateInterface")
##
# Abstract class
# Encodes a state of an MDP
# Pacman: position of every agent, etc

class MDPStateInterface:

	# abstract methods that must be redefined
	def __init__(self) -> None:
		raiseNotDefined()
	def deepCopy(self: TMDPState) -> TMDPState:
		raiseNotDefined()
		return self
	def initFromCopy(self: TMDPState, other: TMDPState) -> None:
		raiseNotDefined()
	def __eq__(self, other: object) -> bool:
		raiseNotDefined()
		if not isinstance(other, MDPStateInterface):
			return NotImplemented
		return True
	def __str__(self) -> str:
		raiseNotDefined()
		return "()"

	# methods that can be redefined if needed
	def consoleStr(self) -> str:
		return str(self)
	def fastReset(self: TMDPState, fastResetData: TMDPState) -> None:
		self.initFromCopy(fastResetData)
	def getFastResetData(self: TMDPState) -> TMDPState:
		fastResetData = self.deepCopy()
		return fastResetData

	# define if you intend to write or read instances from a text file
	def fileStr(self) -> str:
		raiseNotDefined()
		return "s"
	@classmethod
	def fromFileStr(cls: Type[TMDPState], s: str) -> TMDPState:
		raiseNotDefined()
		if s=="":
			raise Exception("parsing error")
		return cls()


TMDPAction = TypeVar("TMDPAction",bound="MDPActionInterface")

##
# Abstract class, Immutable class
# action for controller in an MDP
# Pacman: pacman moves north, east, etc

class MDPActionInterface(util.MiniConsoleInterface):

	# abstract methods that must be redefined
	def __init__(self, infoStr: str) -> None:
		raiseNotDefined()
		self.infoStr = infoStr
	def deepCopy(self: TMDPAction) -> TMDPAction:
		raiseNotDefined()
		return self
	def __hash__(self) -> int:
		raiseNotDefined()
		return 0
	def __eq__(self, other: object) -> bool:
		raiseNotDefined()
		if not isinstance(other, MDPActionInterface):
			return NotImplemented
		return True
	def __str__(self) -> str:
		raiseNotDefined()
		return "("+"infoStr:"+str(self.infoStr)+")"

	# methods that can be redefined if needed
	def consoleStr(self) -> str:
		return str(self)
	def miniConsoleStr(self) -> str:
		return self.consoleStr()

	# define if you intend to write or read instances from a text file
	def fileStr(self) -> str:
		raiseNotDefined()
		return "a"
	@classmethod
	def fromFileStr(cls: Type[TMDPAction], s: str) -> TMDPAction:
		raiseNotDefined()
		if s == "":
			raise Exception("parsing error")
		return cls("")


TMDPStochasticAction = TypeVar("TMDPStochasticAction",bound="MDPStochasticActionInterface")
##
# Abstract class, Immutable class
# action for the environment in an MDP
# pacman: ghost1 moves east and ghost 2 moves north, etc
class MDPStochasticActionInterface(util.MiniConsoleInterface):

	# abstract methods that must be redefined
	def __init__(self, infoStr: str) -> None:
		raiseNotDefined()
		self.infoStr = infoStr
	def deepCopy(self: TMDPStochasticAction) -> TMDPStochasticAction:
		raiseNotDefined()
		return self
	def __hash__(self) -> int:
		raiseNotDefined()
		return 0
	def __eq__(self, other: object) -> bool:
		raiseNotDefined()
		if not isinstance(other, MDPStochasticActionInterface):
			return NotImplemented
		return True
	def __str__(self) -> str:
		raiseNotDefined()
		return "("+"infoStr:"+str(self.infoStr)+")"

	# methods that can be redefined if needed
	def consoleStr(self) -> str:
		return str(self)
	def miniConsoleStr(self) -> str:
		return self.consoleStr()

	# define if you intend to write or read instances from a text file
	def fileStr(self) -> str:
		raiseNotDefined()
		return "a"
	@classmethod
	def fromFileStr(cls: Type[TMDPStochasticAction], s: str) -> TMDPStochasticAction:
		raiseNotDefined()
		if s == "":
			raise Exception("parsing error")
		return cls("")

##
# Immutable class
# encodes a pair (action,stochastic action)
class MDPTransition(Generic[TMDPAction, TMDPStochasticAction]):

	STR_SEPARATOR: str = " "
	FILE_SEPARATOR: str = "\n~\n"
	def __init__(self, mdpAction: TMDPAction, mdpStochasticAction: TMDPStochasticAction) -> None:
		self.mdpAction = mdpAction # an MDPAction instance
		self.mdpStochasticAction = mdpStochasticAction # an MDPStochasticAction instance

	def deepCopy(self) -> "MDPTransition[TMDPAction, TMDPStochasticAction]":
		return MDPTransition(self.mdpAction.deepCopy(),self.mdpStochasticAction.deepCopy())
	def __hash__(self) -> int:
		return hash((self.mdpAction, self.mdpStochasticAction))
	def __eq__(self, other: object) -> bool:
		if not isinstance(other, MDPTransition):
			return NotImplemented
		return self._eq(other)
	def _eq(self, other: "MDPTransition[TMDPAction, TMDPStochasticAction]") -> bool:
		return (self.mdpAction == other.mdpAction) and (self.mdpStochasticAction == other.mdpStochasticAction)

	def __str__(self) -> str:
		return "(mdpAction:"+str(self.mdpAction)+", mdpStochasticAction:" + str(self.mdpStochasticAction) + ")"

	def consoleStr(self) -> str:
		return self.mdpAction.consoleStr()+MDPTransition.STR_SEPARATOR+self.mdpStochasticAction.consoleStr()

	def fileStr(self) -> str:
		return self.mdpAction.fileStr()+MDPTransition.FILE_SEPARATOR+self.mdpStochasticAction.fileStr()

##
# Abstract class
# Encodes a path in the MDP, as an initial state and a sequence of transitions. Also contains a sequence of lists of predicates, one for each state visited along the path
class MDPPath(Generic[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction]):

	FILE_PREFIX: str = "Initial state:\n"
	FILE_SEPARATOR: str = "\nTransitions:\n"
	FILE_LIST_SEPARATOR: str = "\n;\n"
	FILE_PREDICATE_SEPARATOR: str = "\nPredicates:\n"
	def __init__(self, mdpInitialState: TMDPState, mdpTransitionsSequence: List[MDPTransition[TMDPAction, TMDPStochasticAction]], mdpPredicatesSequence: List[List[TMDPPredicate]]) -> None:
		self.mdpInitialState = mdpInitialState # an MDPState instance
		self.mdpTransitionsSequence = mdpTransitionsSequence # a list such that self.mdpTransitionsSequence[i] contains an MDPTransition instance
		self.mdpPredicatesSequence = mdpPredicatesSequence  # a list such that self.mdpPredicatesSequence[i] contains an mdpPredicates list
	def deepCopy(self) -> "MDPPath[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction]":
		return MDPPath(self.mdpInitialState.deepCopy(),[mdpTransition.deepCopy() for mdpTransition in self.mdpTransitionsSequence], [[mdpPredicate.deepCopy() for mdpPredicate in mdpPredicates] for mdpPredicates in self.mdpPredicatesSequence])
	def __str__(self) -> str:
		return "(mdpInitialState:"+str(self.mdpInitialState)+", mdpTransitionsSequence:" + "["+"\n".join( [str(mdpTransition) for mdpTransition in self.mdpTransitionsSequence] ) +"]"+", mdpPredicatesSequence:" + "["+"\n".join( [" ".join([str(mdpPredicate) for mdpPredicate in mdpPredicates ]) for mdpPredicates in self.mdpPredicatesSequence] ) +"]" + ")"
	def __getitem__(self, item):
		return self.mdpTransitionsSequence[item]
	def consoleStr(self) -> str:
		if len(self.mdpPredicatesSequence) != len(self.mdpTransitionsSequence)+1:
			raise Exception("bad length")
		r= ["{"+" ".join([mdpPredicate.consoleStr() for mdpPredicate in self.mdpPredicatesSequence[0]])+"}"]
		for i in range(len(self.mdpTransitionsSequence)):
			r.append(self.mdpTransitionsSequence[i].consoleStr()+' {'+" ".join([mdpPredicate.consoleStr() for mdpPredicate in self.mdpPredicatesSequence[i+1]])+'}')
		return "state:"+self.mdpInitialState.consoleStr()+"->"+"[\n"+"\n".join(r)+"\n]"
		# return "state:"+self.mdpInitialState.consoleStr()+"->"+"[\n"+"\n".join( [mdpTransition.consoleStr() for mdpTransition in self.mdpTransitionsSequence] )+"\n]" #+"[\n"+"\n".join( [ " ".join([mdpPredicate.consoleStr() for mdpPredicate in mdpPredicates]) for mdpPredicates in self.mdpPredicatesSequence] )+"\n]"

	def transitionsCopy(self) -> "MDPPath[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction]":
		"""
		half-shallow copy of an MDPPath instance. The transitions sequence can be modified independently
		but not the transitions themselves nor the initial state.
		"""
		return MDPPath(self.mdpInitialState,[mdpTransition for mdpTransition in self.mdpTransitionsSequence], [mdpPredicates for mdpPredicates in self.mdpPredicatesSequence])
	def lastConsoleStr(self) -> str:
		if len(self.mdpPredicatesSequence) != len(self.mdpTransitionsSequence)+1:
			raise Exception("bad length")
		if len(self.mdpTransitionsSequence) == 0:
			return '{'+" ".join([mdpPredicate.consoleStr() for mdpPredicate in self.mdpPredicatesSequence[-1]])+'}'
		else:
			return self.mdpTransitionsSequence[-1].consoleStr()+' {'+" ".join([mdpPredicate.consoleStr() for mdpPredicate in self.mdpPredicatesSequence[-1]])+'}'
	def lastPredicatesConsoleStr(self) -> str:
		if len(self.mdpPredicatesSequence) != len(self.mdpTransitionsSequence)+1:
			raise Exception("bad length")
		return '{'+" ".join([mdpPredicate.consoleStr() for mdpPredicate in self.mdpPredicatesSequence[-1]])+'}'
	def _subConsoleStr(self, i: int, j: int) -> str:
		if len(self.mdpPredicatesSequence) != len(self.mdpTransitionsSequence)+1:
			raise Exception("bad length")
		r1=""
		if i!=0:
			r1=str(i)+":"
		r2=""
		if j!=self.length():
			r2=":"+str(j)
		r= ["{"+" ".join([mdpPredicate.consoleStr() for mdpPredicate in self.mdpPredicatesSequence[i]])+"}"]
		for x in range(i,j):
			r.append(self.mdpTransitionsSequence[x].consoleStr()+' {'+" ".join([mdpPredicate.consoleStr() for mdpPredicate in self.mdpPredicatesSequence[x+1]])+"}")
		return r1+"["+" | ".join(r)+"]"+r2 #+"\n"+r1+"["+" | ".join( [ " ".join([mdpPredicate.consoleStr() for mdpPredicate in mdpPredicates]) for mdpPredicates in self.mdpPredicatesSequence[i:j+1]] )+"]"+r2
		# return r1+"["+" | ".join( [mdpTransition.consoleStr() for mdpTransition in self.mdpTransitionsSequence[i:j]] )+"]"+r2 #+"\n"+r1+"["+" | ".join( [ " ".join([mdpPredicate.consoleStr() for mdpPredicate in mdpPredicates]) for mdpPredicates in self.mdpPredicatesSequence[i:j+1]] )+"]"+r2
	def suffixConsoleStr(self, depth: int) -> str:
		return self._subConsoleStr(depth,self.length())

	def length(self) -> int:
		return len(self.mdpTransitionsSequence)
	def append(self, mdpTransition: MDPTransition[TMDPAction, TMDPStochasticAction], mdpPredicates: List[TMDPPredicate]) -> None:
		self.mdpTransitionsSequence.append(mdpTransition)
		self.mdpPredicatesSequence.append(mdpPredicates)

	def fastReset(self, i: int) -> None:
		self.mdpTransitionsSequence=self.mdpTransitionsSequence[:i]
		self.mdpPredicatesSequence=self.mdpPredicatesSequence[:i+1]

	def fileStr(self) -> str:
		return MDPPath.FILE_PREFIX+self.mdpInitialState.fileStr()+MDPPath.FILE_SEPARATOR+MDPPath.FILE_LIST_SEPARATOR.join( [mdpTransition.fileStr() for mdpTransition in self.mdpTransitionsSequence])+MDPPath.FILE_PREDICATE_SEPARATOR+MDPPath.FILE_LIST_SEPARATOR.join([ " ".join([mdpPredicate.fileStr() for mdpPredicate in mdpPredicates]) for mdpPredicates in self.mdpPredicatesSequence])
	def initFileStr(self) -> str:
		return self.mdpInitialState.fileStr()+MDPPath.FILE_SEPARATOR
	# def lastFileStr(self) -> str: TODO predicates?
	# 	l=self.length()
	# 	if l==0:
	# 		raise Exception("empty sequence")
	# 	elif l==1:
	# 		return self.mdpTransitionsSequence[-1].fileStr()
	# 	else:
	# 		return MDPPath.FILE_LIST_SEPARATOR+self.mdpTransitionsSequence[-1].fileStr()


def isTerminalStr(isTerminal: bool) -> str:
	if isTerminal:
		return "Terminal"
	else:
		return ""

def discountFactorStr(discountFactor: float) -> str:
	if discountFactor!=1:
		return "discount:"+str(discountFactor)
	else:
		return ""

##
# Abstract class
# Contains a path in the MDP, its last state, total reward, if the path is terminal, the current discount factor at the last state
# All of these things can be derived from the path by executing the sequence of actions on the MDP
class MDPExecution(Generic[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction]):

	FILE_SEPARATOR1: str = "\nEnd state:\n"
	FILE_SEPARATOR2: str = "\nPath reward:\n"
	FILE_SEPARATOR3: str = "\nIs terminal:\n"
	FILE_SEPARATOR4: str = "\nDiscount factor:\n"

	def __init__(self, mdpPath: MDPPath[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction], mdpEndState: TMDPState, mdpPathReward: float, isTerminal: bool, discountFactor: float) -> None:
		self.mdpPath = mdpPath # an MDPPath instance
		self.mdpEndState = mdpEndState # an MDPState instance
		self.mdpPathReward = mdpPathReward # reward obtained along the path, including terminal reward if terminal
		self.isTerminal = isTerminal # a boolean that is True if the path is terminal
		self.discountFactor = discountFactor # the current discount factor
	def deepCopy(self) -> "MDPExecution[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction]":
		return MDPExecution(self.mdpPath.deepCopy(), self.mdpEndState.deepCopy(), self.mdpPathReward, self.isTerminal, self.discountFactor)
	def __str__(self) -> str:
		return "(mdpPath:" + str(self.mdpPath)+", mdpEndState:" + str(self.mdpEndState)+", mdpPathReward:" + str(self.mdpPathReward)+", isTerminal:" + str(self.isTerminal)+", discountFactor:" + str(self.discountFactor) + ")"
	def __getitem__(self, item):
		return self.mdpPath[item]
	def consoleStr(self) -> str:
		return self.mdpPath.consoleStr()+"->"+self.stateConsoleStr()
	def executionCopy(self) -> "MDPExecution[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction]":
		"""
		half-shallow copy of an MDPExecution instance. Can be used to run independent executions on the same MDP.
		"""
		return MDPExecution(self.mdpPath.transitionsCopy(), self.mdpEndState.deepCopy(), self.mdpPathReward, self.isTerminal, self.discountFactor)
	def lastConsoleStr(self) -> str:
		return self.mdpPath.lastConsoleStr()+"->"+self.stateConsoleStr()
	def suffixConsoleStr(self, depth:int) -> str:
		return self.mdpPath.suffixConsoleStr(depth)+"->"+self.stateConsoleStr()
	def stateConsoleStr(self) -> str:
		return "state:"+self.mdpEndState.consoleStr()+" "+self.mdpPath.lastPredicatesConsoleStr()+" reward:"+str(self.mdpPathReward)+" "+isTerminalStr(self.isTerminal)+" "+ discountFactorStr(self.discountFactor)

	def fileStr(self) -> str:
		return self.mdpPath.fileStr()+MDPExecution.FILE_SEPARATOR1+self.mdpEndState.fileStr()+MDPExecution.FILE_SEPARATOR2+str(self.mdpPathReward)+MDPExecution.FILE_SEPARATOR3+str(self.isTerminal)+MDPExecution.FILE_SEPARATOR4+str(self.discountFactor)

	def length(self) -> int:
		return self.mdpPath.length()
	def append(self, mdpTransition: MDPTransition[TMDPAction, TMDPStochasticAction], mdpPredicates: List[TMDPPredicate]) -> None:
#		if self.isTerminal:
#			raise Exception("appending to terminal path")
		self.mdpPath.append(mdpTransition, mdpPredicates)

	def fastReset(self, fastResetData: Tuple[int, TMDPState, float, bool, float]) -> None:
		length,fastResetDataState,mdpPathReward,isTerminal,discountFactor=fastResetData
		self.mdpPath.fastReset(length)
		self.mdpEndState.fastReset(fastResetDataState)
		self.mdpPathReward = mdpPathReward
		self.isTerminal = isTerminal
		self.discountFactor = discountFactor
	def getFastResetData(self) -> Tuple[int, TMDPState, float, bool, float]:
		return (self.length(),self.mdpEndState.getFastResetData(),self.mdpPathReward,self.isTerminal,self.discountFactor)

TMDPOperations = TypeVar("TMDPOperations",bound="MDPOperationsInterface") 

##
# Abstract class
# atomic operations of the MDP: draw next state at random, available actions, etc
# Pacman: contains information about the layout, movement rules, etc
# abstract methods that must be redefined
class MDPOperationsInterface(Generic[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction]):
	def __init__(self, discountFactor: float) -> None:
		raiseNotDefined()
		self.discountFactor = discountFactor
	def deepCopy(self: TMDPOperations) -> TMDPOperations:
		raiseNotDefined()
		return self
	def __str__(self) -> str:
		raiseNotDefined()
		return "(discountFactor:"+str(self.discountFactor)+")"

	def applyTransitionOnState(self, mdpState: TMDPState, mdpTransition: MDPTransition[TMDPAction, TMDPStochasticAction]) -> float:
		raiseNotDefined()
		mdpReward = 0
		# make changes to mdpState according to mdpTransition, and return the reward of this transition
		return mdpReward
	def getStochasticActions(self, mdpState: TMDPState, mdpAction: TMDPAction) -> TMDPStochasticAction:
		raiseNotDefined()
		# returns the set of available stochastic actions according to the support of the MDP's distributions
		raise Exception("undefined")
		# mdpStochasticAction = MDPStochasticActionInterface("")
		# return mdpStochasticAction
	def drawStochasticAction(self, mdpState: TMDPState, mdpAction: TMDPAction, quietInfoStr: bool) -> TMDPStochasticAction:
		raiseNotDefined()
		# draw a stochastic action according to the MDP's distributions
		raise Exception("undefined")
		# mdpStochasticAction = MDPStochasticActionInterface("")
		# return mdpStochasticAction
	def getLegalActions(self, mdpState: TMDPState) -> List[TMDPAction]:
		raiseNotDefined()
		legalActions: List[TMDPAction] = []
		# legalActions.append(MDPActionInterface(""))
		return legalActions

	# methods that can be redefined if needed
	def consoleStr(self) -> str:
		return str(self)
	def replayConsoleStr(self, mdpState: TMDPState) -> str:
		return str(self)
	def getAllPredicates(self) -> List[TMDPPredicate]:
		mdpPredicates: List[TMDPPredicate] = []
		# list all predicates available, true or false
		# mdpPredicates.append(MDPPredicateInterface())
		return mdpPredicates
	def getPredicates(self, mdpState: TMDPState) -> List[TMDPPredicate]:
		mdpPredicates: List[TMDPPredicate] = []
		# list all predicates that hold on mdpState
		# mdpPredicates.append(MDPPredicateInterface())
		return mdpPredicates
	def isExecutionTerminal(self, mdpExecution: MDPExecution[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction]) -> bool:
		# Redefine this method if you want a notion of terminal state/path
		isTerminal = False
		# is the execution over? (terminal state reached, horizon reached, etc)
		return isTerminal
	def getTerminalReward(self, mdpExecution: MDPExecution[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction]) -> float:
		# Redefine this method if you want a notion of terminal reward
		# penalty for losing for exemple
		mdpTerminalReward=0
		# reward for terminal states
		return mdpTerminalReward * mdpExecution.discountFactor

	# define if you intend to write or read instances from a text file
	def fileStr(self) -> str:
		raiseNotDefined()
		return "d"+str(self.discountFactor)
	@classmethod
	def fromFileStr(cls: Type[TMDPOperations], s: str) -> TMDPOperations:
		raiseNotDefined()
		if s=="":
			raise Exception("parsing error")
		if s[:1]!="d":
			raise Exception("bad str provided:"+s)
		discountFactor=float(s[1:])
		return cls(discountFactor)
