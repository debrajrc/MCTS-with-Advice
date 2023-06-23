import stormpy
import stormpy.simulator
import random, json, sys

SEED = 12345678 # change this value to get different paths

options = stormpy.BuilderOptions()
options.set_build_choice_labels()
options.set_build_state_valuations()

def niceStr(state):
	stateDict = json.loads(str(state))
	return str(stateDict)
	# s = f"Time of the day: {stateDict['timeOfTheDay']} \t"
	# s += f"Token: {stateDict['token']} \n"
	# passengerList = []
	# for i in range(0,2):
	# 	if stateDict[f'c{i}_in'] == 1:
	# 		passengerList.append(i) 
	# s += f"Taxi : ({stateDict['xt']},{stateDict['yt']})\t (passengers: {passengerList})\n"
	# for i in range(0,2):
	# 	s += f"Client {i} : ({stateDict[f'xs_c{i}']},{stateDict[f'ys_c{i}']}) -- ({stateDict[f'xc_c{i}']},{stateDict[f'yc_c{i}']}) --> ({stateDict[f'xd_c{i}']},{stateDict[f'yd_c{i}']}) \t (remaining time: {stateDict[f'totalWaiting_c{i}']}) \n"
	# return s

def simulate(path):
	prism_program = stormpy.parse_prism_program(path)
	# model = stormpy.build_sparse_model_with_options(prism_program, options)
	# print(model)

	# simulator = stormpy.simulator.create_simulator(model,seed=SEED)
	simulator = stormpy.simulator.create_simulator(prism_program, seed=SEED)
	assert simulator is not None
	simulator.set_action_mode(stormpy.simulator.SimulatorActionMode.GLOBAL_NAMES)
	simulator.set_observation_mode(stormpy.simulator.SimulatorObservationMode.PROGRAM_LEVEL)
	paths = []
	for m in range(1): # number of paths
		path = []
		state, reward, labels = simulator.restart()
		path = [niceStr(state)]
		for n in range(300): #  number of steps
			actions = simulator.available_actions()
			select_action = random.randint(0,len(actions)-1)
			path.append(f"\n--{actions[select_action]}-->\n\n")
			state, reward, labels = simulator.step(actions[select_action])
			path.append(f"{niceStr(state)}\n reward: {reward[0]}\t labels: {labels}\n")
			if simulator.is_done():
				break
		paths.append(path)
	# for path in paths:
		print("".join(path))

if __name__ == "__main__":
	path = sys.argv[1]
	simulate(path)

