"""interactive script to show neighbors of a given word in a graph"""

import sys
from pathlib import Path
from utils import graph_file_to_list_of_lists, read_vocab


if len(sys.argv) < 3:
    print("Usage: python show_neighbors.py <embedding name> <graph type>")
    sys.exit(1)
    
embeddings = sys.argv[1]
graph_type = sys.argv[2]

data_dir = Path(f"data/{embeddings}")

word_to_idx, idx_to_word = read_vocab(data_dir / "vocab.txt")
        
graph = graph_file_to_list_of_lists(data_dir / "outputs" / graph_type)
        
print("graph loaded")

while True:
    word = input("Enter a word: ")
    if word == "":
        break
    
    if word not in word_to_idx and not word.isdigit():
        print("Word not found")
        continue
    
    if word.isdigit():
        idx = int(word)
        if idx >= len(idx_to_word):
            print("Index out of range")
            continue
        word = idx_to_word[idx]
        print(f"Word {idx}: {word}")
    else:
        idx = word_to_idx[word]
        
    neighbors = graph[idx]
        
    print(f"Neighbors of {word}:")
    for neighbor in neighbors:
        print(idx_to_word[neighbor])
        
    print()
        
    