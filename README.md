# MCTSAdvice2
MCTS with advice

## Requirements
- Python 3.8 or higher
- [stormpy](https://moves-rwth.github.io/stormpy/)
## Example
``python3 main.py examples/traffic/traffic.yml``


## TODO
- Traffic example (generation of prism files from layouts)
- Two prism files: one for the whole MDP ([example](examples/traffic/traffic.nm)) and one for the advice (TODO)
- MCTS options in a config file ([example](examples/traffic/traffic.yml))
- Advice methods defined in a separate python file ([example](examples/traffic/traffic.py))
- Configuration file for advice