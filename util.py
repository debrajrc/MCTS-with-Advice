# TODO: null

import sys,os
import inspect
import heapq, random

from typing import TypeVar, Type, Any, Optional, Sequence, List, Tuple, Dict, Union, Generic, NoReturn

def raiseNotDefined() -> None:
	fileName = inspect.stack()[1][1]
	line = inspect.stack()[1][2]
	method = inspect.stack()[1][3]

	print("*** Method not implemented: %s at line %s of %s" % (method, line, fileName))
	sys.exit(1)

class NoMoveException(Exception):
	pass

def mkdir(dirPath):
	if not os.path.isdir(dirPath):
		# print('The directory',dirPath,'is not present. Creating a new one.')
		try:
			os.makedirs(dirPath)
		except:
			pass
	else:
		pass
		# print('The directory',dirPath,'is present.')

X = TypeVar("X")
class Counter(Dict[X, float]):
	"""
	A counter keeps track of counts for a set of keys.
	"""
	def __getitem__(self, idx: X) -> float:
		self.setdefault(idx, 0.0)
		return dict.__getitem__(self, idx)

	def incrementAll(self, keys: Sequence[X], count: float) -> None:
		"""
		Increments all elements of keys by the same count.

		>>> a = Counter()
		>>> a.incrementAll(['one','two', 'three'], 1)
		>>> a['one']
		1
		>>> a['two']
		1
		"""
		for key in keys:
			self[key] += count

	def argMax(self) -> List[X]:
		"""
		Returns the keys with the highest value.
		"""
		if len(self.keys()) == 0: return []
		all = self.items()
		values = [x[1] for x in all]
		r=[]
		valMax=max(values)
		for k,v in self.items():
			if v==valMax:
				r.append(k)
		# maxIndex = values.index(max(values))
		# return all[maxIndex][0]
		return r

	def sortedKeys(self) -> List[X]:
		"""
		Returns a list of keys sorted by their values.  Keys
		with the highest values will appear first.

		>>> a = Counter()
		>>> a['first'] = -2
		>>> a['second'] = 4
		>>> a['third'] = 1
		>>> a.sortedKeys()
		['second', 'third', 'first']
		"""
		# sortedItems = list(self.items())
		# compare = lambda x, y:  sign(y[1] - x[1])
		# sortedItems.sort(cmp=compare)
		# sortedItems.sort(key=lambda x: -x[1])
		def sortingFunction(x: Tuple[X, float]) -> float:
			return -x[1]
		sortedItems=sorted(self.items(),key=sortingFunction)
		# sortedItems=sorted(self.items(),key=lambda x: -x[1]) # type: ignore
		return [x[0] for x in sortedItems]

	def totalCount(self) -> float:
		"""
		Returns the sum of counts for all keys.
		"""
		return sum(self.values())

	def normalize(self) -> None:
		"""
		Edits the counter such that the total count of all
		keys sums to 1.  The ratio of counts for all keys
		will remain the same. Note that normalizing an empty
		Counter will result in an error.
		"""
		total = float(self.totalCount())
		if total == 0 or total == 1: return
		for key in self.keys():
			self[key] = self[key] / total

	def divideAll(self, divisor: float) -> None:
		"""
		Divides all counts by divisor
		"""
		divisor = float(divisor)
		for key in self:
			self[key] /= divisor

	def copy(self) -> "Counter[X]":
		"""
		Returns a copy of the counter
		"""
		return Counter(dict.copy(self))

	def __mul__(self, y: "Counter[X]" ) -> float:
		"""
		Multiplying two counters gives the dot product of their vectors where
		each unique label is a vector element.

		>>> a = Counter()
		>>> b = Counter()
		>>> a['first'] = -2
		>>> a['second'] = 4
		>>> b['first'] = 3
		>>> b['second'] = 5
		>>> a['third'] = 1.5
		>>> a['fourth'] = 2.5
		>>> a * b
		14
		"""
		sum = 0.0
		x = self
		if len(x) > len(y):
			x,y = y,x
		for key in x:
			if key not in y:
				continue
			sum += x[key] * y[key]
		return sum

	def __radd__(self, y: "Counter[X]") -> None:
		"""
		Adding another counter to a counter increments the current counter
		by the values stored in the second counter.

		>>> a = Counter()
		>>> b = Counter()
		>>> a['first'] = -2
		>>> a['second'] = 4
		>>> b['first'] = 3
		>>> b['third'] = 1
		>>> a += b
		>>> a['first']
		1
		"""
		for key, value in y.items():
			self[key] += value

	def __add__( self, y: "Counter[X]" ) -> "Counter[X]":
		"""
		Adding two counters gives a counter with the union of all keys and
		counts of the second added to counts of the first.

		>>> a = Counter()
		>>> b = Counter()
		>>> a['first'] = -2
		>>> a['second'] = 4
		>>> b['first'] = 3
		>>> b['third'] = 1
		>>> (a + b)['first']
		1
		"""
		addend: "Counter[X]" = Counter()
		for key in self:
			if key in y:
				addend[key] = self[key] + y[key]
			else:
				addend[key] = self[key]
		for key in y:
			if key in self:
				continue
			addend[key] = y[key]
		return addend

	def __sub__( self, y: "Counter[X]" ) -> "Counter[X]":
		"""
		Subtracting a counter from another gives a counter with the union of all keys and
		counts of the second subtracted from counts of the first.

		>>> a = Counter()
		>>> b = Counter()
		>>> a['first'] = -2
		>>> a['second'] = 4
		>>> b['first'] = 3
		>>> b['third'] = 1
		>>> (a - b)['first']
		-5
		"""
		addend: "Counter[X]" = Counter()
		for key in self:
			if key in y:
				addend[key] = self[key] - y[key]
			else:
				addend[key] = self[key]
		for key in y:
			if key in self:
				continue
			addend[key] = -1 * y[key]
		return addend

class StrFloatCounter(Counter[str]):
	def __str__(self) -> str:
		return '{' + ' '.join([str(k)+":"+"{:.2f}".format(self[k]) for k in self.sortedKeys()]) + '}'

class MiniConsoleInterface:
	def miniConsoleStr(self) -> str:
		raiseNotDefined()
		return ""
TMiniConsoleInterface = TypeVar("TMiniConsoleInterface",bound=MiniConsoleInterface)

class ConsoleStrFloatCounter(Counter[TMiniConsoleInterface]):
	def __str__(self) -> str:
		return '{' + ' '.join([k.miniConsoleStr()+":"+"{:.2f}".format(self[k]) for k in self.sortedKeys()]) + '}'


def normalize(vector: List[float]) ->  List[float]:
	"""
	normalize a vector or counter by dividing each value by the sum of all values
	"""
	s = float(sum(vector))
	if s == 0: return vector
	return [el / s for el in vector]

def sample(distribution: Counter[X]) -> X:
	# items = sorted(distribution.items())
	items = distribution.items()
	distributionL = [i[1] for i in items]
	values = [i[0] for i in items]
	if sum(distributionL) != 1:
		distributionL = normalize(distributionL)
	choice = random.random()
	i, total= 0, distributionL[0]
	while choice > total:
		i += 1
		total += distributionL[i]
	return values[i]

def chooseFromDistribution(distribution: Counter[X]) -> X:
	"Takes a counter and samples"
	return sample(distribution)
