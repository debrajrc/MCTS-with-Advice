import sys, os

CURRENTPWD = os.path.dirname(os.path.abspath(__file__))

sys.path.append(os.path.join(CURRENTPWD, '../src'))

from playPacmanStorm import replayResults
from adviceMCTS.stormMdpClasses import *
from pacmanPrism import createEngine

LAYOUTS_DIR = CURRENTPWD+os.sep+'layouts'

layoutFile = LAYOUTS_DIR+os.sep+"halfClassic.lay"
engine = createEngine(layoutFile)
prismFile = engine.createPrismFile()
prismSimulator = prismToSimulator(prismFile)
mdpOperations = MDPOperations(prismSimulator, prismFile, str, 1)

def replay(fileName,layoutName):
	f = open(fileName, 'r')
	f.readline()
	f.readline()
	engineList = readResults(f,mdpOperations)
	f.close()
	replayResults(engineList, layoutName,0.05)

if __name__ == "__main__":
	replay(sys.argv[1],LAYOUTS_DIR+os.sep+'halfClassic.lay')
