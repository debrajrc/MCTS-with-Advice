from simulationClasses import MDPPathAdviceInterface, MDPActionAdviceInterface, MDPStateScoreInterface, MDPExecutionEngine
from stormMdpClasses import MDPState, MDPOperations, MDPAction
import math
import random
import sys
import time
import curses
import os

from typing import TypeVar, Type, Any, Optional, Sequence, List, Tuple, Dict, Union, Generic, NoReturn

import util
import stormpy
from util import raiseNotDefined, NoMoveException
from mdpClasses import *
dirname = os.path.dirname(__file__)

sys.path.append(dirname)
from trafficPrism import TaxiEngine

# terminal reward


class MDPStateScore(MDPStateScoreInterface):
    def getScore(self, executionEngine: MDPExecutionEngine) -> float:
        return 0


class MDPStateScoreDistance(MDPStateScoreInterface):

    def __init__(self):
        p = TaxiEngine.fromFile(dirname+"/layouts/_10x10_0_spawn.lay")
        (X, Y, number_of_clients, fuel_position, walls) = p.getLayoutDescription()
        self.X = X
        self.Y = Y
        self.number_of_clients = number_of_clients
        self.fuel_position = fuel_position
        self.walls = walls

    def gridDistance(self, posList):
        infty = self.X*self.Y+1
        r = [[infty for j in range(self.Y)] for i in range(self.X)]
        queue = []
        maxd = infty
        for ((x, y), direction) in posList:
            r[x][y] = 0
            # queue.append((x,y,0))
            maxd = 0
            (xr, yr) = (x, y)
            # if direction == 0:
            #     (xr, yr) = (-1, -1)
            # if direction == 1:
            #     yr -= 1
            # elif direction == 2:
            #     yr += 1
            # if direction == 3:
            #     xr += 1
            # elif direction == 4:
            #     xr -= 1
            for xx, yy in [(x-1, y), (x, y-1), (x+1, y), (x, y+1)]:
                if xx >= 0 and xx < self.X and yy >= 0 and yy < self.Y and (not self.walls[xx][yy]) and r[xx][yy] > 1 and (xx, yy) != (xr, yr):
                    r[xx][yy] = 1
                    queue.append((xx, yy, 1))
                    maxd = 1
        while len(queue) > 0:
            x, y, d = queue.pop(0)
            for xx, yy in [(x-1, y), (x, y-1), (x+1, y), (x, y+1)]:
                if xx >= 0 and xx < self.X and yy >= 0 and yy < self.Y and (not self.walls[xx][yy]) and r[xx][yy] > d+1:
                    r[xx][yy] = d+1
                    queue.append((xx, yy, d+1))
                    if d+1 > maxd:
                        maxd = d+1

        return (r, maxd)

    def getScore(self, executionEngine):
        endState = executionEngine.mdpEndState()
        stateDescription = executionEngine.mdpOperations.stateDescription(
            endState)
        coef = 0.5
        inValuation = 0
        fuelValuation = 0

        taxiPos = stateDescription['xt'], stateDescription['yt']

        taxiDistance, maxTaxiDist = self.gridDistance([(taxiPos, 0)])
        isOccupied = False
        for k in range(self.number_of_clients):
            if stateDescription[f'c{k}_in'] == 1:
                destPos = (stateDescription[f'xd_c{k}'], stateDescription[f'yd_c{k}'])
                d = taxiDistance[destPos[0]][destPos[1]]
                inValuation += 1/(1+d)
                isOccupied = True
        if not isOccupied:
            for k in range(self.number_of_clients):
                clientPos = (stateDescription[f'xs_c{k}'], stateDescription[f'ys_c{k}'])
                d = taxiDistance[clientPos[0]][clientPos[1]]
                inValuation += 1/(10*(1+d))
        # else:
        #     inValuation += 100/(1+maxTaxiDist)
        # inValuation /= self.number_of_clients
        if stateDescription['totalFuel'] <= 10:
            if not isOccupied:
                d = taxiDistance[self.fuel_position[0]][self.fuel_position[1]]
                fuelValuation += 100/(1+d)

        return coef*inValuation + (1-coef)*fuelValuation


# selection advice
class MDPSafeActionAdvice(MDPActionAdviceInterface):

    def getMDPActionAdvice(self, mdpState: MDPState, mdpOperations: MDPOperations, quietInfoStr: bool) -> tuple[list[MDPAction]]:
        choices = mdpOperations.getLegalActions(mdpState)
        if not quietInfoStr:
            for mdpAction in choices:
                if mdpAction.infoStr != '':
                    mdpAction.infoStr += '#'
                mdpAction.infoStr += 'AdviceFull'
        return choices, choices


class MDPFullActionAdvice(MDPActionAdviceInterface[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction]):
    """!
    A trivial advice that allow everything
    """
    TMDPOperations = TypeVar(
        "TMDPOperations", bound=MDPOperationsInterface[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction])

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


# path advice
class MDPSafePathAdvice(MDPPathAdviceInterface):

    def isValidPath(self, mdpExecutionEngine: MDPExecutionEngine) -> bool:
        mdpPredicatesList = mdpExecutionEngine.mdpOperations.getPredicates(
            mdpExecutionEngine.mdpEndState())
        for predicate in mdpPredicatesList:
            if predicate.name == "Unsafe":
                return False
        return True


class MDPFullPathAdvice(MDPPathAdviceInterface[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction]):
    def isValidPath(self, mdpExecutionEngine: MDPExecutionEngine[TMDPPredicate, TMDPState, TMDPAction, TMDPStochasticAction]) -> bool:
        return True


class MDPStormActionAdvice(MDPActionAdviceInterface):
    """! action advice that returns action over a given threshold according to Storm
    @params depth Unfording of the MDP with  horizon depth is created and Storm is used to find the probability of staying safe
    @params taxiEngine
    @params threshold

    @returns choices set of actions
    """

    def __init__(self):
        # depth = 3
        # self.depth = depth
        taxiEngine = TaxiEngine.fromFile(dirname+"/layouts/_10x10_0_spawn.lay")
        self.taxiEngine = taxiEngine
        self.threshold = 0.9  # TODO ????

    def _getMDPActionAdviceInSubset(self, mdpActions, mdpState, mdpOperations):
        """
        The Agent will receive an MDPOperations and
        must return a subset of a set of legal mdpActions
        """
        if len(mdpActions) == 1:
            return mdpActions
        choices = []
        # print("\nAvailable actions:", [
        #       action.action for action in mdpActions])  # debuging
        stateDescription = mdpOperations.stateDescription(mdpState)
        # print(niceStr(stateDescription)) # debuging
        agentInfo = self.taxiEngine.getInfo(stateDescription)
        # print("Agent info:", agentInfo)  # debuging
        actionValues = []
        d = {1: "North", 2: "South", 3: "East", 4: "West"}
        # TODO why "1" for the start ? The "0" is considered at root ?
        for taxi_first_action in range(1, 5):
            # print(niceStr(stateDescription))
            # print(self.taxiEngine.printLayout(stateDescription))  # debuging
            util.mkdir(dirname+"/temp")
            # print("==> ", d[taxi_first_action])
            prismFile = dirname+"/temp/" + \
                str(os.getpid()) + '_'+str(taxi_first_action)+'_advice.nm'

            p = self.taxiEngine.newEngine(
                agentInfo, taxi_first_action, prismFile)

            p.createPrismFilefFromGrids(safety=True)

            prism_program = stormpy.parse_prism_program(prismFile)
            formula_str = "Pmax=? [(G ! unsafe)]"
            properties = stormpy.parse_properties(formula_str, prism_program)
            model = stormpy.build_model(prism_program, properties)
            initial_state = model.states[0]
            if 'deadlock' in list(model.labels_state(initial_state)):
                print(
                    f"{d[taxi_first_action] } {initial_state}==>{list(model.labels_state(initial_state))}")
                value = 0  # (float('nan'))
            else:
                result = stormpy.model_checking(
                    model, properties[0], only_initial_states=True)
                value = result.at(initial_state)
            actionValues.append(value)
            os.remove(prismFile)
        # print(actionValues) # # debuging
        maxValue = max(actionValues)
        if maxValue == 1:  # if there are always safe actions choose always safe actions
            threshold = 1
        else:
            threshold = self.threshold
        for action in mdpActions:
            try:
                actionId = ['North', 'South', 'East',
                            'West'].index(action.action)
            except:
                raise Exception(
                    "multiple actions"+str([str(mdpAction) for mdpAction in mdpActions]))
            if actionValues[actionId] >= threshold*maxValue:
                choices.append(action)
        if len(choices) == 0:
            return mdpActions
        # print("Best actions:", [action.action for action in choices], "\n") # # debuging
        return choices


# a function to print the states nice
def niceStr(stateDict: dict) -> str:
    s = f"Time of the day: {stateDict['timeOfTheDay']} \t"
    s += f"Level of fuel: {stateDict['totalFuel']} \t"
    s += f"Counter of jam: {stateDict['jamCounter']} \t"
    s += f"Token: {stateDict['token']} \n"
    passengerList = []
    for i in range(0, 2):
        if stateDict[f'c{i}_in'] == 1:
            passengerList.append(i)
    s += f"Taxi : ({stateDict['xt']},{stateDict['yt']})\t (passengers: {passengerList})\n"
    for i in range(0, 2):
        s += f"Client {i} : ({stateDict[f'xs_c{i}']},{stateDict[f'ys_c{i}']}) -- ({stateDict[f'xc_c{i}']},{stateDict[f'yc_c{i}']}) --> ({stateDict[f'xd_c{i}']},{stateDict[f'yd_c{i}']}) \t (remaining time: {stateDict[f'totalWaiting_c{i}']}) \n"
    return s


def niceStrGrid(stateDict: dict) -> str:
    p = TaxiEngine.fromFile(dirname+"/layouts/_10x10_0_spawn.lay")
    str = niceStr(stateDict) + p.printLayout(stateDict)
    return str
