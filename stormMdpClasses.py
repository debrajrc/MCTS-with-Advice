"""! @brief A class using stormpy and mdpClasses"""

from typing import TypeVar, Type, Any, Optional, Sequence, List, Tuple, Dict, Union, Generic, NoReturn

from mdpClasses import *
import stormpy
import stormpy.simulator
from stormpy.storage import BitVector
from simulationClasses import *
import json

def prismToSimulator(prismFile):
	"""! Given a prism file, creates a stormpy simulator

	@param prismFile Location of the Prism file
	@return  a stormpy.simulator.PrismSimulator object
	"""
	prism_program = stormpy.parse_prism_program(prismFile)
	option = stormpy.core.BuilderOptions()
	option.set_build_state_valuations()
	option.set_build_choice_labels()
	prismSimulator = stormpy.simulator.create_simulator(prism_program)
	prismSimulator.set_action_mode(stormpy.simulator.SimulatorActionMode.GLOBAL_NAMES)
	return prismSimulator


class MDPPredicate(MDPPredicateInterface):

	# methods that must be redefined
	def __init__(self,name: str):
		self.name = name

	def deepCopy(self) -> "MDPPredicate":
		return MDPPredicate(self.name)

	def initFromCopy(self, other: "MDPPredicate") -> None:
		self.name = other.name

	def __str__(self) -> str:
		return "(name:"+str(self.name)+")"

	# methods that can be redefined
	def consoleStr(self) -> str:
		return str(self.name)

	# define if you intend to write or read instances from a text file
	def fileStr(self) -> str:
		return str(self.name)
	@classmethod
	def fromFileStr(cls, s: str) -> "MDPPredicate":
		# if s=="":
			# raise Exception("parsing error") ## NOTE: why this?
		name=s
		return cls(name)

class MDPState(MDPStateInterface):
	# methods that must be redefined
	def __init__(self, bitVector) -> None:
		self.bitVector=bitVector

	def deepCopy(self) -> "MDPState":
		return MDPState(self.bitVector)

	def initFromCopy(self, other: "MDPState") -> None:
		self.bitVector = other.bitVector

	def __str__(self) -> str:
		return "(bitVector:"+self.bitVector.store_as_string()+")"

	def __eq__(self, other: object) -> bool:
		if not isinstance(other, MDPState):
			return NotImplemented
		return (self.bitVector == other.bitVector)

	# methods that can be redefined
	def consoleStr(self) -> str:
		return self.bitVector.store_as_string()

	# define if you intend to write or read instances from a text file
	def fileStr(self) -> str:
		return self.bitVector.store_as_string()

	@classmethod
	def fromFileStr(cls, s:str) -> "MDPState":
		bitVector = BitVector().load_from_string(s)
		return cls(bitVector)


class MDPAction(MDPActionInterface):
	# Immutable class
	# methods that must be redefined
	def __init__(self,action: str,infoStr: str="") -> None:
		self.action=action
		self.infoStr=infoStr

	def deepCopy(self) -> "MDPAction":
		return MDPAction(self.action,self.infoStr)

	def __hash__(self) -> int:
		return hash(self.action)

	def __eq__(self, other: object) -> bool:
		if not isinstance(other, MDPAction):
			return NotImplemented
		return (self.action == other.action)

	def __str__(self) -> str:
		return "(action:"+str(self.action)+",infoStr:"+str(self.infoStr)+")"

	# methods that can be redefined
	def consoleStr(self) -> str:
		return str(self.action)+("#"+str(self.infoStr) if self.infoStr!="" else "")

	def miniConsoleStr(self) -> str:
		return str(self.action)

	# define if you intend to write or read instances from a text file
	def fileStr(self) -> str:
		return str(self.action)+("\n#"+str(self.infoStr) if self.infoStr!="" else "")
	@classmethod
	def fromFileStr(cls, s: str) -> "MDPAction":
		# if s=="":
		#     raise Exception("parsing error")
		splits=s.split("\n#")
		action=splits[0]
		infoStr=""
		for i in range(1,len(splits)):
			infoStr+=splits[i]
		return cls(action,infoStr)

class MDPStochasticAction(MDPStochasticActionInterface):
	# Immutable class
	# methods that must be redefined
	def __init__(self,bitVector, reward:float, infoStr: str="") -> None:
		self.bitVector=bitVector
		self.reward=reward
		self.infoStr=infoStr

	def deepCopy(self) -> "MDPStochasticAction":
		return MDPStochasticAction(self.bitVector,self.reward,self.infoStr)

	def __hash__(self) -> int:
		return hash(self.bitVector)

	def __eq__(self, other: object) -> bool:
		if not isinstance(other, MDPStochasticAction):
			return NotImplemented
		return (self.bitVector == other.bitVector)

	def __str__(self) -> str:
		return "(bitVector:"+str(self.bitVector.store_as_string())+"reward:"+str(self.reward)+",infoStr:"+str(self.infoStr)+")"

	# methods that can be redefined
	def consoleStr(self) -> str:
		return str(self.bitVector.store_as_string())+("#"+str(self.infoStr) if self.infoStr!="" else "")

	def miniConsoleStr(self) -> str:
		return str(self.bitVector.store_as_string())

	# define if you intend to write or read instances from a text file
	def fileStr(self) -> str:
		return self.bitVector.store_as_string()+("\n#"+str(self.reward))+("\n#"+str(self.infoStr) if self.infoStr!="" else "")
	@classmethod
	def fromFileStr(cls, s: str) -> "MDPStochasticAction":
		if s=="":
			raise Exception("parsing error")
		splits=s.split("\n#")
		bitVectorStr=splits[0]
		bitVector=BitVector().load_from_string(bitVectorStr)
		reward=float(splits[1])
		if len(splits) > 2:
			infoStr=splits[2]
		else:
			infoStr=""
		return cls(bitVector, reward, infoStr)

	def toState(self) -> "MDPState":
		return MDPState(self.bitVector)

class MDPOperations(MDPOperationsInterface[MDPPredicate, MDPState, MDPAction, MDPStochasticAction]):
	FILE_SEPARATOR: str = "\nParameters\n"

	# methods that must be redefined
	def __init__(self, prismSimulator, prismFile, stateStrFunction = str, discountFactor = 1) -> None:
		self.prismSimulator=prismSimulator
		self.prismFile=prismFile
		self.discountFactor=discountFactor
		self.stateStrFunction=stateStrFunction

	def updateStrFunction(self, stateStrFunction):
		self.stateStrFunction=stateStrFunction

	def deepCopy(self) -> "MDPOperations":
		return MDPOperations(self.prismSimulator,self.prismFile,self.discountFactor,self.stateStrFunction) # NOTE: this is not deepcopy, as I cannot deepcopy prism simulator

	# def __str__(self) -> str:
	#     return "(walls:\n"+gridStr(self.walls)+",holes:\n"+gridStr(self.holes)+",targets:\n"+gridStr(self.targets)+",discountFactor:"+str(self.discountFactor)+")"

	def applyTransitionOnState(self, mdpState: MDPState, mdpTransition: MDPTransition[MDPAction, MDPStochasticAction]) -> float:
		mdpState.bitVector=mdpTransition.mdpStochasticAction.bitVector
		mdpReward=mdpTransition.mdpStochasticAction.reward
		return mdpReward

	def drawStochasticAction(self, mdpState: MDPState, mdpAction: MDPAction, quietInfoStr: bool) -> MDPStochasticAction:
		currentBitVector = self.prismSimulator._get_current_state()
		stateBitVector = mdpState.bitVector
		if currentBitVector != stateBitVector:
			self.prismSimulator.restart(stateBitVector)
		actionName = mdpAction.action
		self.prismSimulator.step(actionName)
		newBitVector = self.prismSimulator._get_current_state()
		reward = self.prismSimulator._report_rewards()
		if len(reward) > 1:
			raise Exception ("multiple reward")
		mdpStochasticAction = MDPStochasticAction(newBitVector,reward[0],"")
		return mdpStochasticAction

	def getLegalActions(self, mdpState: MDPState) -> List[MDPAction]:
		currentBitVector = self.prismSimulator._get_current_state()
		stateBitVector = mdpState.bitVector
		if currentBitVector != stateBitVector:
			self.prismSimulator.restart(stateBitVector)
		actions = self.prismSimulator.available_actions()
		legalActions=[]
		for label in actions:
			action = MDPAction(label,"")
			if action not in legalActions:
				legalActions.append(action)
		return legalActions

	def getAllPredicates(self) -> List[MDPPredicate]:
		mdpPredicates: List[MDPPredicate] = []
		labels = [label.name for label in self.prismSimulator._program.prism_program.labels]
		mdpPredicates = [MDPPredicate(label) for label in labels]
		return mdpPredicates

	def getPredicates(self, mdpState: MDPState) -> List[MDPPredicate]:
		currentBitVector = self.prismSimulator._get_current_state()
		stateBitVector = mdpState.bitVector
		if currentBitVector != stateBitVector:
			self.prismSimulator.restart(stateBitVector)
		labels = self.prismSimulator._report_labels()
		mdpPredicates: List[MDPPredicate] = [MDPPredicate(label) for label in labels]
		# if currentBitVector != stateBitVector:
		#     self.prismSimulator.restart(currentBitVector)
		return mdpPredicates

	def isExecutionTerminal(self, mdpExecution: MDPExecution[MDPPredicate, MDPState, MDPAction, MDPStochasticAction]) -> bool:
		currentBitVector = self.prismSimulator._get_current_state()
		mdpState = mdpExecution.mdpEndState
		stateBitVector = mdpState.bitVector
		if currentBitVector != stateBitVector:
			self.prismSimulator.restart(stateBitVector)
		isTerminal = self.prismSimulator.is_done()
		return isTerminal

	def getTerminalReward(self, mdpExecution: MDPExecution[MDPPredicate, MDPState, MDPAction, MDPStochasticAction]) -> float:
		return 0

	def stateDescription(self,mdpState): # returns a dict # works also with stochastic actions
		currentBitVector = self.prismSimulator._get_current_state()
		stateBitVector = mdpState.bitVector
		if currentBitVector != stateBitVector:
			self.prismSimulator.restart(stateBitVector)
		stateDescription = json.loads(str(self.prismSimulator._report_state()))
		return stateDescription

	def replayConsoleStr(self,mdpState):
		stateDescription = self.stateDescription(mdpState)
		return self.stateStrFunction(stateDescription)

	# define if you intend to write or read instances from a text file
	def fileStr(self) -> str:
		return str (self.prismFile) + MDPOperations.FILE_SEPARATOR + "\ndiscount " + str(self.discountFactor)

	@classmethod
	def fromFileStr(cls, s: str, stateStrFunction=str) -> "MDPOperations":
		split=s.split(MDPOperations.FILE_SEPARATOR)
		if len(split)!=2:
			raise Exception("parsing error mdp "+s)
		prismFile = split[0]
		prismSimulator = prismToSimulator(prismFile)
		params = split[1].split('\n')
		if params[1][:8]!="discount":
			raise Exception("bad str provided",params[1])
		discountFactor=float(params[1][8:])
		return cls(prismSimulator, prismFile, stateStrFunction, discountFactor)

def runResults(engineList: List[MDPExecutionEngine[MDPPredicate, MDPState, MDPAction, MDPStochasticAction]], cursesDelay:float = 0.0, quiet: bool = True, prettyConsole: bool = False) -> Any:
	"""! Given a prism file and two formula, gives the conditional distance

	@param engineList list of saved executions
	@param cursesDelay a float that adds delay between printing states (looks cool if prettyConsole is True)
	@param quiet if True, does not print debugging info
	@param prettyConsole if True, print state using predefined function instead of str (using MDPOperations.stateStrFunction)
	"""
	print("[== running replay engines")
	n = 0
	for mdpExecutionEngine in engineList:
		print(f"---------\ngame#{n}\n---------")
		cursesScr = None
		if prettyConsole:
			cursesScr = curses.initscr()
			curses.noecho()
			curses.cbreak()
			# cursesDelay = 0.01#min(0.5,max(0.05,60.0/mdpExecutionEngine.length(ignoreNonDecisionStates=False)))

		optionsReplayEngine=OptionsReplayEngine(quiet=quiet, printEachStep=False, printCompact=False, cursesScr=cursesScr, cursesDelay=cursesDelay)
		mdpReplayEngine = MDPReplayEngine(mdpExecutionEngine.mdpOperations,mdpExecutionEngine.mdpPath(),options=optionsReplayEngine)
		# TODO: instead of try-except, use the wrapper function in curses
		try:
			while not mdpReplayEngine.isTerminal():
				mdpReplayEngine.advanceReplay()
			# if (mdpReplayEngine.mdpExecutionEngine.isTerminal() != mdpExecutionEngine.isTerminal()) or (mdpReplayEngine.mdpExecutionEngine.mdpPathReward() != mdpExecutionEngine.mdpPathReward()) or (mdpReplayEngine.mdpExecutionEngine.mdpEndState() != mdpExecutionEngine.mdpEndState()):
			#     raise Exception("bad replay")
			if prettyConsole:
				curses.endwin()
		except BaseException as error:
			if prettyConsole:
				curses.endwin()
			raise Exception(str(error))
		n+=1
		print(f"Score\t\t{mdpExecutionEngine.mdpPathReward()}")
		resultString = "Result\t\t"
		result = mdpExecutionEngine.result()
		if len(result) == 0:
			resultString += "__"
		else:
			for p in result:
				resultString+=p.name+" "
		print(resultString)
		print(f"Length\t\t{mdpExecutionEngine.length(ignoreNonDecisionStates=True)}")
		print('---------')
	print("==] done\n")


def runGamesWithMCTS(stateStrFunction,prismFile,**kwargs):
	"""! Given a prism file, runs MCTS with it

	@param stateStrFunction custom defined function to print states (using MDPOperations.stateStrFunction)
	@param prismFile location to a prism file
	@param **kwargs arguments for MCTS (See simulationClasses.MDPMCTSTraceEngine.runMCTSTrace())
	"""
	discount = kwargs['discount']
	kwargs.pop('discount')
	prismSimulator = prismToSimulator(prismFile)
	mdp = MDPOperations(prismSimulator,prismFile,stateStrFunction,discount)
	bitVector = prismSimulator._get_current_state()
	initState = MDPState(bitVector)
	labels = prismSimulator._report_labels()
	initPredicates = [MDPPredicate(label) for label in labels]
	traceEngine: MDPMCTSTraceEngine[MDPPredicate, MDPState, MDPAction, MDPStochasticAction] = MDPMCTSTraceEngine()
	results = traceEngine.runMCTSTrace(mdpState=initState, mdpPredicates=initPredicates, mdpOperations=mdp,**kwargs)

	return(results)

# def getFirstMCTSValues(stateStrFunction,prismFile=None,**kwargs):
#
# 	if prismFile is None:
# 		if 'prismFile' not in kwargs:
# 			if 'layout' in kwargs:
# 				raise Exception("no prismFile arg provided")
# 		else:
# 			prismFile = kwargs['prismFile']
# 	else:
# 		if 'prismFile' not in kwargs:
# 			pass
# 		elif kwargs['prismFile'] == None:
# 			kwargs.pop('prismFile')
# 		else:
# 			raise Exception("more than one prismFile arg provided")
# 	layoutFile = kwargs['layout']
# 	kwargs.pop('layout')
# 	kwargs.pop('replay')
# 	discount = kwargs['discount']
# 	kwargs.pop('discount')
# 	prismSimulator = prismToSimulator(prismFile)
# 	mdp = MDPOperations(prismSimulator,prismFile,stateStrFunction,discount)
# 	bitVector = prismSimulator._get_current_state()
# 	initState = MDPState(bitVector)
# 	labels = prismSimulator._report_labels()
# 	initPredicates = [MDPPredicate(label) for label in labels]
# 	traceEngine: MDPMCTSTraceEngine[MDPPredicate, MDPState, MDPAction, MDPStochasticAction] = MDPMCTSTraceEngine()
# 	results = traceEngine.runMCTSTrace(mdpState=initState, mdpPredicates=initPredicates, mdpOperations=mdp,**kwargs)
# 	return(results)

##
# Parsers for simulation classes

def MDPTransitionfromFileStr(s:str) -> MDPTransition[MDPAction,MDPStochasticAction]:
	splits=s.split(MDPTransition.FILE_SEPARATOR)
	s1=splits[0]
	s2=""
	for i in range(1,len(splits)):
		s2+=splits[i]
	return MDPTransition(MDPAction.fromFileStr(s1),MDPStochasticAction.fromFileStr(s2))

def MDPPathfromFileStr(s:str) -> MDPPath[MDPPredicate, MDPState, MDPAction, MDPStochasticAction]:
	if MDPPath.FILE_PREFIX==s[:len(MDPPath.FILE_PREFIX)]:
		s=s[len(MDPPath.FILE_PREFIX):]
	else:
		raise Exception("bad prefix in parsing path")
	splits=s.split(MDPPath.FILE_SEPARATOR)
	s1=splits[0]
	s2=""
	for i in range(1,len(splits)):
		s2+=splits[i]
	splits1 = s2.split(MDPPath.FILE_PREDICATE_SEPARATOR)
	if len(splits1) != 2:
		raise Exception("Parse error")
	splits2=splits1[0].split(MDPPath.FILE_LIST_SEPARATOR)
	splits3=splits1[1].split(MDPPath.FILE_LIST_SEPARATOR)
	return MDPPath(MDPState.fromFileStr(s1),[MDPTransitionfromFileStr(ss) for ss in splits2],[[MDPPredicate.fromFileStr(sss) for sss in ss.split(" ")] for ss in splits3])

def MDPExecutionfromFileStr(s:str) -> MDPExecution[MDPPredicate, MDPState, MDPAction, MDPStochasticAction]:
	splits=s.split(MDPExecution.FILE_SEPARATOR1)
	if len(splits) != 2:
		raise Exception("Parse error")
	mdpPath = MDPPathfromFileStr(splits[0])
	splits=splits[1].split(MDPExecution.FILE_SEPARATOR2)
	if len(splits) != 2:
		raise Exception("Parse error")
	mdpEndState = MDPState.fromFileStr(splits[0])
	splits=splits[1].split(MDPExecution.FILE_SEPARATOR3)
	if len(splits) != 2:
		raise Exception("Parse error")
	mdpPathReward = float(splits[0])
	splits=splits[1].split(MDPExecution.FILE_SEPARATOR4)
	if len(splits) != 2 or not (splits[0] == "True" or splits[0] == "False"):
		raise Exception("Parse error")
	isTerminal = (splits[0] == "True")
	discountFactor = float(splits[1])
	return MDPExecution(mdpPath, mdpEndState, mdpPathReward, isTerminal, discountFactor)

def readResults(file: Any, mdpOperations = None) -> List[MDPExecutionEngine[MDPPredicate, MDPState, MDPAction, MDPStochasticAction]]:
	FILE_PREFIX = 'MDP:\n'
	s = file.read()
	s = s.split('\nTraceLists:\n')
	if len(s) < 2:
		return []
	if len(s) != 2:
		raise Exception("Parsing error")
	so = s[0]
	if FILE_PREFIX==so[:len(FILE_PREFIX)]:
		so = so[len(FILE_PREFIX):]
	else:
		raise Exception("bad prefix in parsing path")
	if mdpOperations == None: # I am adding this option so I can create one mdp and give it as input to read multiple results
		mdpOperations = MDPOperations.fromFileStr(so) # type: Any
	st = s[1]
	TRACE_SEPERATOR = '\nTrace:\n'
	st = st.split(TRACE_SEPERATOR)
	engineList = []
	for t in st:
		st = t.split('\nnumSim:\n')
		if len(st) != 2:
			print(st)
			raise Exception("parser errror")
		mdpExecution = MDPExecutionfromFileStr(st[0])
		numSim = int(st[1])
		if numSim != 1:
			raise Exception("numSim not 1")
		# mdpPath = MDPPathfromFileStr(t)
		engine = MDPExecutionEngine(mdpOperations, mdpExecution,0) # type: MDPExecutionEngine[MDPPredicate, MDPState, MDPAction, MDPStochasticAction]
		engineList.append(engine)
	return engineList

##
# Save to file

def printResults(traces: List[Tuple[MDPExecutionEngine[MDPPredicate, MDPState, MDPAction, MDPStochasticAction],float,int]], file: Any = sys.stdout) -> None:
	TRACE_SEPERATOR = '\nTrace:\n'
	if len(traces)<1 or len(traces[0])<1:
		raise Exception("empty trace")
	print ('MDP:\n'+traces[0][0].mdpOperations.fileStr()+'\nTraceLists:\n'+TRACE_SEPERATOR.join([r[0].mdpExecution.fileStr()+'\nnumSim:\n'+str(r[2]) for r in traces]),file=file)


class MDPStateScore(MDPStateScoreInterface):
	def getScore(self, executionEngine):
		return 0

class MDPSafeActionAdvice( MDPActionAdviceInterface):

	def getMDPActionAdvice(self, mdpState, mdpOperations, quietInfoStr: bool):
		choices = mdpOperations.getLegalActions(mdpState)
		if not quietInfoStr:
			for mdpAction in choices:
				if mdpAction.infoStr != '':
					mdpAction.infoStr += '#'
				mdpAction.infoStr += 'AdviceFull'
		return choices, choices

def zeroScoreFunction(**kwargs):
	return 0

class MDPNonLossPathAdvice(MDPPathAdviceInterface[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction]):
	def isValidPath(self, mdpExecutionEngine: MDPExecutionEngine[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction]) -> bool:
		mdpPredicatesList = mdpExecutionEngine.mdpOperations.getPredicates(mdpExecutionEngine.mdpEndState())
		for predicate in mdpPredicatesList:
			if predicate.name == "Loss":
				return False
		return True

if __name__ == "__main__":
	pass
