import stormpy, os, sys
import gc

CURRENTPWD = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(CURRENTPWD, '../src'))

import util
from adviceMCTS.conditionalMinDist import getDistValue

DEBUG = False

TEMP_DIR = 'tempFiles'+os.sep+'prism'
util.mkdir(TEMP_DIR)

def createPrismFilefFromGrids(walls,holes,targets,position,initAction):
	height = len(walls)
	width = len(walls[0])

	fname = TEMP_DIR+os.sep+str(os.getpid())+f'_{initAction}.nm'
	f = open(fname,'w+')

	print ('mdp\n', file = f)

	formulaWin = 'formula win = '
	formulaLoss = 'formula loss = '

	formulaNorth = 'formula north = '
	formulaSouth = 'formula south = '
	formulaEast = 'formula east = '
	formulaWest = 'formula west = '

	print (f'const int initAction = {initAction};', file = f)
	print('global initStep : bool init true;', file = f)

	for i in range(height):
		for j in range(width):
			print (f'const int x{i}_{j} = {i};', file = f)
			print (f'const int y{i}_{j} = {j};', file = f)
			print (f'const int w{i}_{j} = {int(walls[i][j])};', file = f)
			print (f'const int h{i}_{j} = {int(holes[i][j])};', file = f)
			print (f'const int t{i}_{j} = {int(targets[i][j])};', file = f)
			if i > 0:
				formulaNorth += f'(x{i}_{j} = x & y{i}_{j} = y & w{i-1}_{j} = 0) | '
			if i < height-1:
				formulaSouth += f'(x{i}_{j} = x & y{i}_{j} = y & w{i+1}_{j} = 0) | '
			if j < width-1:
				formulaEast += f'(x{i}_{j} = x & y{i}_{j} = y & w{i}_{j+1} = 0) | '
			if j > 0:
				formulaWest += f'(x{i}_{j} = x & y{i}_{j} = y & w{i}_{j-1} = 0) | '
			formulaWin += f'(x{i}_{j} = x & y{i}_{j} = y & t{i}_{j} = 1) | '
			formulaLoss += f'(x{i}_{j} = x & y{i}_{j} = y & h{i}_{j} = 1) | '

	print('global end : bool init false;\n', file = f)

	formulaNorth = formulaNorth[:-3]+';\n'
	formulaSouth = formulaSouth[:-3]+';\n'
	formulaEast = formulaEast[:-3]+';\n'
	formulaWest = formulaWest[:-3]+';\n'
	formulaWin = formulaWin[:-3]+';\n'
	formulaLoss = formulaLoss[:-3]+';\n'
	print(formulaNorth, file = f)
	print(formulaSouth, file = f)
	print(formulaEast, file = f)
	print(formulaWest, file = f)
	print(formulaWin, file = f)
	print(formulaLoss, file = f)

	print(f'formula northInt = north?1:0;\n', file = f)
	print(f'formula southInt = south?1:0;\n', file = f)
	print(f'formula eastInt = east?1:0;\n', file = f)
	print(f'formula westInt = west?1:0;\n', file = f)
	print(f'formula numDirNorth = westInt + northInt + eastInt;\n', file = f)
	print(f'formula numDirSouth = southInt + eastInt + westInt;\n', file = f)
	print(f'formula numDirEast = northInt + southInt + eastInt;\n', file = f)
	print(f'formula numDirWest = northInt + southInt + westInt;\n', file = f)

	print('\nmodule robot\n', file = f)

	print(f'x : [0..{height-1}] init {position[0]};', file = f)
	print(f'y : [0..{width-1}] init {position[1]};\n', file = f)

	print("[North] ((initAction = 0 | initAction = 3 | initStep = false) & north & !win & !loss & !end) -> (10 * northInt)/(numDirNorth+9): (x'=x-1) & (initStep' = false) + (1 * eastInt)/(numDirNorth+9): (y'=y+1) & (initStep' = false) + (1 * westInt)/(numDirNorth+9): (y'=y-1) & (initStep' = false);", file = f)
	print("[South] ((initAction = 0 | initAction = 4 | initStep = false) & south  & !win & !loss & !end) -> (10 * southInt)/(numDirSouth+9): (x'=x+1) & (initStep' = false) + (1 * eastInt)/(numDirSouth+9): (y'=y+1) & (initStep' = false) + (1 * westInt)/(numDirSouth+9): (y'=y-1) & (initStep' = false);", file = f)
	print("[East] ((initAction = 0 | initAction = 1 | initStep = false) & east  & !win & !loss & !end) -> (1 * northInt)/(numDirEast+9): (x'=x-1) & (initStep' = false) + (1 * southInt)/(numDirEast+9): (x'=x+1) & (initStep' = false) + (10 * eastInt)/(numDirEast+9): (y'=y+1) & (initStep' = false);", file = f)
	print("[West] ((initAction = 0 | initAction = 2 | initStep = false) & west & !win & !loss & !end) -> (1 * northInt)/(numDirWest+9): (x'=x-1) & (initStep' = false) + (1 * southInt)/(numDirWest+9): (x'=x+1) & (initStep' = false) + (10 * westInt)/(numDirWest+9): (y'=y-1) & (initStep' = false);", file = f)
	print ("[] (win | loss) -> 1 : (end' = true);", file = f)
	print('\nendmodule\n', file = f)

	print('rewards', file = f)
	print('[East] true : -1;',file=f)
	print('[West] true: -1;',file=f)
	print('[North] true: -1;',file=f)
	print('[South] true: -1;',file=f)
	print ('(win & !end) : 100;', file = f)
	print ('(loss & !end) : -100;', file = f)
	print('endrewards\n', file = f)

	print('label "win" = win;', file = f)
	print('label "loss" = loss;', file = f)
	f.close()
	return(fname)

def getValue(prismFile,formula_str = "Pmax=? [F win]"):
	prism_program = stormpy.parse_prism_program(prismFile)
	properties = stormpy.parse_properties(formula_str, prism_program)
	model = stormpy.build_model(prism_program, properties)
	# print(model)
	result = stormpy.model_checking(model, properties[0],only_initial_states=True)
	# assert result.result_for_all_states
	initial_state = model.initial_states[0]
	value = result.at(initial_state)
	# del prism_program
	del model
	# del result
	# del initial_state
	gc.collect()
	return(value)

def getAllValuesFromGrids(walls,holes,targets,position):
	values = []
	for i in range(1,5):
		prismFile = createPrismFilefFromGrids(walls,holes,targets,position, i)
		value = getValue(prismFile)
		values.append(value)
		# if DEBUG == True:
		# 	print(prismFile)
		# else:
		# 	os.system("rm "+prismFile)
	return values

def getAllDistValuesFromGrids(walls,holes,targets,position):
	values = []
	for i in range(1,5):
		prismFile = createPrismFilefFromGrids(walls,holes,targets,position, i)
		formula1 = "Pmax=? [F win]"
		formula2 = "Tmin=? [F win]"
		value = getDistValue(prismFile,formula1,formula2)
		values.append(value)
		# if DEBUG == True:
		# 	print(prismFile)
		# else:
		# 	os.system("rm "+prismFile)
	return values

if __name__ == "__main__":
	prismFile = "prism20081_3.nm"
	print(getValue(prismFile))
