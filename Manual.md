## 🍒 Configuration file

The configuration file is a YAML file with the following structure:

```yaml
# basic parameters
verbosity : 
prism file : 
number of games : 
number of steps : 
discount factor : 

# mcts parameters
mcts:
  number of simulations : 
  number of iterations : 
  horizon : 
  mcts constant : 
  alpha : 

# problem specific parameters
other parameters:
  python file : 
  state score : 
  selection advice : 
  selection advice at root : 
  simulation action advice : 
  simulation path advice : 
  print function : 

```

### Basic parameters

- `verbosity` : 0 - 5 determines the amount of information in MCTS printed to the console
- `prism file` : path to the prism file describing the MDP
- `number of games` : number of games to be played
- `number of steps` : number of steps in each game
- `discount factor` : discount factor for the MDP (between 0 and 1; in case of undiscounted reward, use 1.)

### MCTS parameters

- `number of simulations` : number of simulations per iteration
- `number of iterations` : number of iterations
- `horizon` : horizon for the MCTS (one step refers to one controllable action from a state from which multiple actions are available)
- `mcts constant` : constant for the UCT formula (Write as a float, so $\frac{\sqrt{2}}{2}$ can be written as 0.7071067811865475.)
- `alpha` : parameter to adjust terminal reward. If total reward of path is `mdpPathReward`, the MCTS algorithm takes the value of the path as `mdpReward = (1-alpha)*mdpPathReward + alpha*stateScore` where `stateScore` is (user-defined) score of the state at the end of the path.

### Problem specific parameters

The user can define classes to describe the advice in a python file. The user can also define a method to print the state of the MCTS algorithm. The path to this file is given in the configuration file. 

- `python file` : path to the python file containing the advice classes and the print function. Instead of `/` as the separator for directories, use `.`. For example, if the file is in the directory `examples/traffic/` and is called `traffic.py`, the path is `examples.traffic.traffic`.


## 🍒 Advice

The action-based advice can be used in both the selection and the simulation phases of the MCTS algorithm. The path-based advice is used in the simulation phase of the MCTS algorithm.

### Action-based advice

> `Interface class` : `MDPActionAdviceInterface`

An action-based strategy can viewed as a nondeterministic policy. This advice provides the method `_getMDPActionAdviceInSubset(mdpActions, mdpState, mdpOperations)` that takes a set of actions, a state of the MDP and the description of the MDP, and returns a subset of actions.

`stateDescription = mdpOperations.stateDescription(mdpState)` returns a `dict` with the values of the variables in the state. This can be useful to define an action-based advice that depends on these values.

### Path-based advice

> `Interface class` : `MDPPathAdviceInterface`

A path-based advice provides the method `isValidPath(mdpExecutionEngine)` that takes a state of the MDP and the description of the MDP, and returns a Bool.

One way to define path advice is to check the labels (defined in the Prism file) satisfied by the states in the path. Few code snippets that can be useful in the implementation of the method `isValidPath` are:

```python
# get all predicates in the MDP
predicates = mdpExecutionEngine.getAllPredicates()

# get a list of lists of predicates where ith list contains all predicates in the ith state of the path
predicateList = mdpExecutionEngine.getPredicatesSequence()

# get predicates in the end state of the path
mdpPredicatesList = mdpExecutionEngine.mdpOperations.getPredicates(mdpExecutionEngine.mdpEndState())

```
Let `predicates` be the set of predicates in a state. Then `[predicate.name for predicate in predicates]` would return a list of strings which are the labels satisfied by the state.

## 🍒 Score function

> `Interface class` : `MDPStateScoreInterface`

This class provides the method `getScore(executionEngine)` that assigns a score to an excecution. The score function is used in the MCTS algorithm as follows: If the total reward of a path (excecution) is `mdpPathReward`, the MCTS algorithm takes the value of the path as `mdpReward = (1-alpha)*mdpPathReward + alpha*stateScore` where `stateScore` is the score of the state at the end of the path.

Given an executionEngine, the following snippet would return a `dict` with the values of the variables in the end state of the execution:

```python
endState = executionEngine.mdpEndState()
stateDescription = executionEngine.mdpOperations.stateDescription(endState)
```
This can be useful to define a score function that depends on the values of the variables in the end state of the execution.

## 🍒 Print function

After a trace is generated, the trace can be simulated in the console. To do this,
a print function can be defined to print a state in the python file containing the advice classes. The print function takes a `dict` with the values of the state variables in a state of the execution as input and returns a string.

## 🍒 Example (Pac-Man)

The configuration file is [examples/pacman/pacman.yml](examples/pacman/pacman.yml) and the python file containing the advice classes and the print function is [examples/pacman/pacmanAdvice.py](examples/pacman/pacmanAdvice.py).

To run the example, use the following command:

```bash
python3 main.py examples/pacman/pacman.yml
```