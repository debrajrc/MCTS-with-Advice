import stormpy, os
import numpy as np
from pacmanPrismNoFood import *

def getValue(prismFile,quiet=False):
	"""!
	From a prismFile describing finite horizon unfolding of PacMan (ignoring food), find the probability to stay safe
	"""
	prism_program = stormpy.parse_prism_program(prismFile)
	option = stormpy.core.BuilderOptions()
	option.set_build_state_valuations()
	option.set_build_choice_labels()
	formula_str = "Pmax=? [(G ! isLoss)]"
	properties = stormpy.parse_properties(formula_str, prism_program)
	if not quiet:
		print('prismFile :', prismFile)
		print('building model...')
	model = stormpy.build_model(prism_program, properties)
	if not quiet:
		print('model built')
	if not quiet:
		print(model)
	initial_state = model.states[0]
	if 'deadlock' in list(model.labels_state(initial_state)):
		return 0 #(float('nan'))
	if not quiet:
		print('checking model...')
	result = stormpy.model_checking(model, properties[0],only_initial_states=True)
	value = result.at(initial_state)
	if not quiet:
		print(f"Value : {value}")
	return(value)

def getAllActionValues(layoutFile,depth,quiet):
	"""!
	This function returns a list of 4 values.
	@params layoutFile text file descibing Pac-Man layout
	@params depth horizon to unfold for model checking

	@returns
	A list l containing 4 floats. Let D = [East,West,North,South]. l[i] denotes the probability of staying safe for given depth by taking action D[i] and then playing optimally.
	"""
	prismFile = TEMP_DIR+os.sep+layoutFile.split(os.sep)[-1][:-4]+'_'+str(os.getpid())+'.nm'
	height,width,numGhosts,layoutText,agentInfo = readFromFile(layoutFile)
	valueList = []
	for pacmanFirstAction in range(1,5):
		p = pacmanEngine(height,width,numGhosts,layoutText,agentInfo,pacmanFirstAction=pacmanFirstAction,drawDepth =depth, fname=prismFile)
		p.createPrismFile()
		value = getValue(prismFile,quiet=quiet)
		if not quiet:
			print(f"Value for action {['_','East','West','North','South'][pacmanFirstAction]}: {value}")
		valueList.append(value)
	os.remove(prismFile)
	if not quiet:
		print(f"values : {valueList}")
	return(valueList)

def getAllActionValuesArray(layoutFile,depth,quiet):
	"""!
	This function returns a list of 4 values.
	@params layoutFile text file descibing Pac-Man layout
	@params depth horizon to unfold for model checking

	@returns X A tensor of 6 channels (without food) to represent initial state
	@returns Y An array l of length 4. Let D = [East,West,North,South]. l[i] denotes the probability of staying safe for given depth by taking action D[i] and then playing optimally.
	"""
	# print(layoutFile)
	prismFile = TEMP_DIR+os.sep+layoutFile.split(os.sep)[-1][:-4]+'_'+str(os.getpid())+'.nm'
	height,width,numGhosts,layoutText,agentInfo = readFromFile(layoutFile)
	x_created = False
	Y = np.zeros((4))
	for pacmanFirstAction in range(1,5):
		p = pacmanEngine(height,width,numGhosts,layoutText,agentInfo,pacmanFirstAction=pacmanFirstAction,drawDepth =depth, fname=prismFile)
		p.createPrismFile()
		if not x_created:
			X = p.getArray()
			x_created = True
		# print(f"Value for action {['_','East','West','North','South'][pacmanFirstAction]}: {getValue(prismFile,quiet=quiet)}")
		Y[pacmanFirstAction-1] = getValue(prismFile,quiet=quiet)
	os.remove(prismFile)
	return(X,Y)

if __name__ == '__main__':
		layoutFile = sys.argv[1]
		depth = int(sys.argv[2])
		quiet = [False,True][int(sys.argv[3])]
		getAllActionValues(layoutFile,depth,quiet)
		# X,Y = getAllActionValuesArray(layoutFile,depth,quiet)
		# with np.printoptions(threshold=np.inf):
		# 	print(X,Y)
