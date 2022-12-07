# studious-potato
MCTS with advice


# Examples
## Frozen Lake

python3 examples/frozenLake/playFrozenLake.py --nn --nnModel examples/frozenLake/models/mcts.h5 --nnThreshold 0.99 -l examples/frozenLake/layouts/0_10x10_0.lay --replay

python3 examples/frozenLake/playFrozenLake.py --stormDist -l examples/frozenLake/layouts/0_10x10_0.lay --replay

python3 examples/frozenLake/playFrozenLake.py --mcts -l examples/frozenLake/layouts/0_10x10_0.lay --ignoreNonDecisionStates -n 1 -s 100 -iter 50 -sim 10 -H 20 -psT -qm -qs -qi --replay


## Pac-Man

python3 examples/pacman/playPacmanStorm.py --mcts --ActionScoreThreshold 0.9 --ActionScoreThresholdRoot 0.9 --MCTSActionAdvice NN --MCTSActionAdviceRoot NN --MCTSConstant 1000 --MCTSPathAdviceSim NonLoss --MCTSStateScore DistanceOld --ModelActionScore examples/pacman/models/safety.h5 --TransformerActionScore None --ModelActionScoreRoot examples/pacman/models/safety.h5 --TransformerActionScoreRoot None --horizon 10 --layout examples/pacman/layouts/halfClassic.lay --numMCTSIters 20 --numMCTSSims 10 --numTraces 1 --stateScoreWeight 0.5 --steps 701 --ignoreNonDecisionStates -v --replay --replayDelay 0.05

python3 examples/pacman/playPacmanStorm.py --nn --nnModel examples/pacman/models/all.h5 --nnTransformer None -l examples/pacman/layouts/halfClassic.lay --replay -s 2100 --replayDelay 0.1 --nnThreshold 1