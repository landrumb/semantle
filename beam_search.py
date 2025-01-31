"""implementations of natural variants of beam search"""

import numpy as np
from typing import List, Tuple

def dist(a, b):
    """negative inner product, but this could be changed"""
    return -np.dot(a, b)

def beam_search(graph : List[List[int]], vectors: np.ndarray, start : int, query : int, limit : int = 1000) -> Tuple[List[int], List[int]]:
    """standard beam search with no limit; terminates when the end node is reached
    
    In a connected graph, guaranteed to terminate eventually"""
    # beam elements are dist, index
    beam = [(dist(vectors[start], vectors[query]))]
    compared = []
    visited = []
    while beam and len(visited) < limit:
        # get the best element
        _, best = beam.pop(0)
        visited.append(best)
        # if the best element is the query, return the path
        if best == query:
            return visited, compared
        # add the neighbors of the best element to the beam
        for neighbor in graph[best]:
            if neighbor not in compared:
                beam.append((dist(vectors[neighbor], vectors[query]), neighbor))
                compared.append(neighbor)
        # sort the beam
        beam.sort()
        
    return visited, compared # this should probably never be reached
        
def eager_beam_search(graph : List[List[int]], vectors: np.ndarray, start : int, query : int, limit : int = 1000) -> Tuple[List[int], List[int]]:
    """beam search which stops going through the neighbors of a point when something better is found"""
    # beam elements are dist, index
    beam = [(dist(vectors[start], vectors[query]), start)]
    compared = []
    visited = []
    while beam and len(visited) < limit:
        # get the best element
        best_dist, best = beam.pop(0)
        # if the best element is the query, return the path
        if best == query:
            return visited, compared
        # add the neighbors of the best element to the beam
        for neighbor in graph[best]:
            if neighbor not in compared:
                beam.append((dist(vectors[neighbor], vectors[query]), neighbor))
                compared.append(neighbor)
                if dist(vectors[neighbor], vectors[query]) < best_dist:
                    beam.append((best_dist, best))
                    break
        else:
            visited.append(best)
        # sort the beam
        beam.sort()
        
    return visited, compared # this should probably never be reached
        
        
    