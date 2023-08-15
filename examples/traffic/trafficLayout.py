"""
Author      : DESMET Aline 
Matricule   : 000474868 (MA2-INFO)
Description : Generate random layouts
"""

import random
import os
from typing import List, Tuple
import copy

Grid = List[List[bool]]
Position = Tuple[int, int]


def createLayout(number_of_layouts: int, height: int, width: int, number_of_stops: int, fuel_level: int, probability: float = 0.9, directory_layout: str = 'examples' + os.sep + 'traffic'+os.sep + "layouts") -> None:
    try:
        os.mkdir(directory_layout)
    except OSError as error:
        print("{0} was not created as it already exists.".format(directory_layout))
    filename = directory_layout+os.sep+'_'+str(height)+'x'+str(width)+'_'
    for i in range(number_of_layouts):
        file = open(filename+str(i)+"_spawn.lay", "w+")
        print("OK1")

        walls = setWallsPositions(height, width, probability)

        stops, fuel_station, airport, taxi_position = setOtherPositions(
            walls, number_of_stops)
        print("OK2")

        grid_str = f"{height} {width} {number_of_stops} {fuel_level}\n"
        grid_str += transformToStr(walls, stops,
                                   fuel_station, airport, taxi_position)
        print(grid_str, file=file)
        file.close()


def setWallsPositions(height: int, width: int, probability: float = 0.9) -> Grid:

    city_grid = []
    for i in range(height):  # Lines
        row = []
        for j in range(width):  # Columns
            random_probability = random.random()
            (is_upperLower_borders) = i == 0 or i == height-1
            is_side_borders = j == 0 or j == width-1
            if (is_upperLower_borders) or (is_side_borders):
                row.append(True)
            elif random_probability > probability:
                row.append(True)
            else:
                row.append(False)
        city_grid.append(row)
    return city_grid


def setOtherPositions(walls: Grid, number_of_stops: int):
    height, width = len(walls), len(walls[0])
    fuel_station = [[False for j in range(width)] for i in range(height)]
    airport = [[False for j in range(width)] for i in range(height)]
    taxi = [[False for j in range(width)] for i in range(height)]
    stops = [[False for j in range(width)] for i in range(height)]

    stops = setPositions(
        walls, height, width, number_of_stops, stops, stops)

    fuel_station = setPositions(
        walls, height, width, 1, fuel_station, stops[0])[0]

    airport = setPositions(walls, height, width, 1,
                           airport, fuel_station, stops[0])
    taxi = setPositions(walls, height, width, 1, taxi,
                        airport[0], fuel_station, stops[0])[1][0]

    return stops, fuel_station, airport, taxi


def setPositions(walls: Grid, height: int, width: int, number_of_elems: int, position_to_change: Grid, *other_positions: Grid):

    is_available = False
    counter = 0
    coords_list = []
    while not is_available or counter < number_of_elems:
        print("OK3")
        line = random.randint(0, height-1)
        column = random.randint(0, width-1)
        i = 0
        if len(other_positions) > 0:
            is_available = not walls[line][column] and not other_positions[i][line][column]
        else:
            is_available = not walls[line][column]
        while i < len(other_positions)-1 and is_available:
            is_available = not walls[line][column] and not other_positions[i+1][line][column]
            i += 1

        if is_available:
            position_to_change[line][column] = True
            coords_list.append((line, column))
            counter += 1

    return position_to_change, coords_list


def distributeProbabilities(x, max) -> List:
    probabilities = []
    remaining_probability = 1 - max

    probabilities.append(max)
    # Generate random probabilities for x positions
    for i in range(x-1):
        # Generate a random probability between 0 and the remaining probability
        probability = random.uniform(0, remaining_probability)
        probabilities.append(probability)
        remaining_probability -= probability

    # The last position gets the remaining probability
    probabilities.append(remaining_probability)
    return probabilities


def transformToStr(walls: Grid, stops: Grid, fuel_station: Grid, airport: Grid, taxi_position: Position, probability: int = 0.9) -> str:

    grid_str = "\n".join(["".join(["T" if (i == taxi_position[0] and j == taxi_position[1]) else ("%" if walls[i][j] else ("." if stops[0][i][j] else (
        "F" if fuel_station[i][j] else ("A" if airport[0][i][j] else " ")))) for j in range(len(walls[i]))]) for i in range(len(walls))])
    probabilities = distributeProbabilities(len(stops[1]) + 1, 0.5)
    probabilities.sort()
    spawn_probability = probabilities.pop()
    grid_str += f"\n {airport[1][0][0]} {airport[1][0][1]} {spawn_probability}"
    random.shuffle(probabilities)

    for i in range(len(stops[1])):
        grid_str += f"\n {stops[1][i][0]} {stops[1][i][1]} {probabilities[i]}"
    return grid_str


if __name__ == '__main__':
    height, width = 5,5

    createLayout(3, height, width, 2, (height*width)//2)

    pass
