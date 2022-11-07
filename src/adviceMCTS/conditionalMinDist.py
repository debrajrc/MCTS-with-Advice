"""! @brief Distance-optimal strategy for reachability."""

import stormpy, math
import gc

def investigateModel(model):
	for state in model.states:
		actionOffset = 0
		for action in state.actions:
			for transition in action.transitions:
				actionIndex = model.get_choice_index(state,actionOffset)
				print("From state {}, with action {}, with probability {}, go to state {}".format(state,actionIndex, transition.value(),transition.column))
			actionOffset += 1

class ConditionalMinDistEngine():

	def __init__(self,prismFile):
		self.prismProgram = stormpy.parse_prism_program(prismFile)

	def getValue(self,formula): # get the values for each states
		properties = stormpy.parse_properties(formula, self.prismProgram) # formula is of the form "Pmax=?..."
		self.model = stormpy.build_model(self.prismProgram, properties)
		result = stormpy.model_checking(self.model, properties[0])
		self.result = result
		return result

	def getBestActionIds(self,state): # returns list of integers for best actions from a state
		bestActionIds = []
		bestValue = 0  # assuming we are maximizing the probability
		for action in state.actions:
			value = 0
			for transition in action.transitions:
				value += self.result.at(transition.column)*transition.value()
			if value > bestValue:
				bestValue = value
				bestActionIds = [action.id]
			elif value == bestValue:
				bestActionIds += [action.id]
			else:
				pass
		return bestActionIds

	def removeBadStates(self): # create a new model by removing states with 0 value

		# building new transition matrix
		numColumns = self.model.transition_matrix.nr_columns # we need to fix the columns so that storm does not remove disconnected states while building MDP
		builder = stormpy.SparseMatrixBuilder(rows=0, columns=numColumns, entries=0, force_dimensions=True,has_custom_row_grouping=True, row_groups=0)

		numChoices = 0 # keeps track of the index of choices

		# each row group is for a state
		for state in self.model.states:
			builder.new_row_group(numChoices) # add state; create a new group of rows

			bestActions = self.getBestActionIds(state)

			# each row is for a choice: (state, action) pair
			for action in state.actions:
				if self.result.at(state) == 0: # if bad state
					builder.add_next_value(numChoices, state, 1) # self loop at the bad state
					pass
				elif action.id not in bestActions: # if bad action
					builder.add_next_value(numChoices, state, 1) # self-loop : this will work as optimal stategy will not take this transition. I am doing this because otherwise there is an open choice with no transition probabilities associated with it. Then Tmin is not giving correct result. Better way would have been adding an extra sink state. But then I need to change the state labeling manually
				else: # if both state and action are good
					# # calculate the normalization factor : (1 - badProbability)
					# badProbability = 0
					# for transition in action.transitions:
					# 	if self.result.at(transition.column) == 0: # if bad next state
					# 		badProbability += transition.value()
					#
					for transition in action.transitions:
						# if self.result.at(transition.column) == 0: # if bad next state
						# 	pass
						# 	# builder.add_next_value(numChoices, transition.column, 0) # add stochastic transition with probability 0
						# else:
						# print(state, numChoices, transition.column, self.result.at(transition.column))
						newProbability = transition.value()*self.result.at(transition.column)/self.result.at(state)
						if newProbability != 0:
							builder.add_next_value(numChoices, transition.column, newProbability) # add normalized stochastic transition

				numChoices += 1 # even when we are not adding the choices in MDP, we maintain the same index with original model

		transition_matrix = builder.build()

		# keep labels of states
		state_labeling = self.model.labeling

		# not adding reward model
		reward_models = {}

		# build new mdp
		components = stormpy.SparseModelComponents(transition_matrix=transition_matrix, state_labeling=state_labeling,reward_models=reward_models, rate_transitions=False)
		newModel = stormpy.storage.SparseMdp(components)
		self.newModel = newModel
		return (newModel)

	def getBestDistValue(self,formula): # get the expected min distance for each states
		# formula is of the form "Tmin=?..."
		properties = stormpy.parse_properties(formula, self.prismProgram)
		newResult = stormpy.model_checking(self.newModel, properties[0])
		self.newResult = newResult
		return newResult

	def getBestDistActionIds(self,state): # returns list of integers for best actions minimizing expected distance from a state
		bestActionIds = []
		bestValue = math.inf  # assuming we are minimizing the distance
		for action in state.actions:
			if action.id in self.getBestActionIds(state):
				value = 0
				for transition in action.transitions:
					if self.newResult.at(transition.column) == math.inf:
						print(transition.value())
						assert (transition.value() == 0) # bad states are reached with probability 0
					else:
						value += self.newResult.at(transition.column)*transition.value()
				if value < bestValue:
					bestValue = value
					bestActionIds = [action.id]
				elif value == bestValue:
					bestActionIds += [action.id]
				else:
					pass
			else:
				pass
		return bestActionIds

	def process(self, formula1, formula2):
		self.getValue(formula1)
		self.removeBadStates()
		self.getBestDistValue(formula2)

	def getFinalValues(self): # returns probability to reach in original MDP and distance value in new MDP
		oldInitialState = self.model.initial_states[0]
		newInitialState = self.newModel.initial_states[0]
		value = self.result.at(oldInitialState)
		distance = self.newResult.at(newInitialState)
		return value, distance

def getDistValue(prismFile,formula1,formula2):
	"""! Given a prism file and two formula, gives the conditional distance

	@param prismFile Location of the Prism file
	@param formula1 formula with Pmax
	@param formula2 formula with Pmin

	@code
	prismFile = "../examples/prism20081_3.nm"
	formula1 = "Pmax=? [F win]"
	formula2 = "Tmin=? [F win]"
	v = getDistValue(prismFile,formula1,formula2)
	print(v)
	@endcode
	"""
	c = ConditionalMinDistEngine(prismFile)
	c.process(formula1, formula2)
	v = c.getFinalValues()
	del c
	gc.collect()
	return v

if __name__ == "__main__":
	pass
