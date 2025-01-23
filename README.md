# Vector Search as Semantle

This is a clone of the game [Semantle](https://semantle.com/). In Semantle, players guess words and see their semantic similarity to a secret target word. This is a vector search problem, where the vectors are word embeddings, dressed up as a game. Players try to minimize the number of guesses they need to find the target word, much as an index over a set of vectors tries to minimize the number of distance comparisons required to solve a query.

The objective of this project is to evaluate whether people are better at vector search (i.e. who needs the fewest comparisons to find the 1-NN of a query) than popular graph-based methods. 