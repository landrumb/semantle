# Vector Search as Semantle

This is a clone of the game [Semantle](https://semantle.com/). In Semantle, players guess words and see their semantic similarity to a secret target word. This is a vector search problem, where the vectors are word embeddings, dressed up as a game. Players try to minimize the number of guesses they need to find the target word, much as an index over a set of vectors tries to minimize the number of distance comparisons required to solve a query.

The objective of this project is to evaluate whether people are better at vector search (i.e. who needs the fewest comparisons to find the 1-NN of a query) than popular graph-based methods.

## Setup

1. If you don't have it already, get [uv](https://docs.astral.sh/uv/)
2. Clone the submodules: `git submodule update --init --recursive`
3. Symlink a directory you want to store data in to `data/` (e.g. `ln -s /my/data/dir data`)
4. Run the setup script: `./SETUP.sh`

