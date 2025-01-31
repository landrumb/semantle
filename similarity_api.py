from flask import Flask, request, jsonify
from flask_cors import CORS
from pathlib import Path
from utils import fbin_to_numpy, graph_file_to_list_of_lists, read_vocab
import numpy as np
import subprocess

app = Flask(__name__)
CORS(app, supports_credentials=True, resources={r"/*": {"origins": "*"}})  # enable cross-origin requests

MODEL = "word2vec-google-news-300_50000_lowercase"

data_dir = Path("data") / MODEL

if not data_dir.exists():
    subprocess.run(["bash", "remote_setup.sh"], check=True)

print("loading vectors...")
vectors = fbin_to_numpy(data_dir / "base.fbin")
word_to_idx, vocab = read_vocab(data_dir / "vocab.txt")

print("loading graph...")
graph = graph_file_to_list_of_lists(data_dir / "outputs" / "vamana")
bfs_distances = []
with open(data_dir / "outputs" / "vamana_distances.txt") as f:
    for line in f:
        bfs_distances.append(int(line))

@app.route("/get_vocab", methods=["GET"])
def get_vocab():
    """returns the vocabulary list to the client so it can select a target word"""
    return jsonify({"vocab": vocab})

@app.route("/similarity", methods=["POST"])
def get_similarity():
    """returns the pairwise similarities between two sets of words"""
    data = request.json
    word1 = data.get("word1", [])
    word2 = data.get("word2", [])

    if any(word not in vocab for word in word1 + word2):
        return jsonify({"error": "at least one word not in vocabulary"}), 400

    word1_indices = [word_to_idx[word] for word in word1]
    word2_indices = [word_to_idx[word] for word in word2]
    similarity = np.dot(vectors[word1_indices], vectors[word2_indices].T)
    similarity = [[float(x) for x in row] for row in similarity]
    return jsonify({"similarities": similarity})
    

@app.route("/top_k", methods=["POST"])
def get_top_k():
    """returns the top k most similar words to a target word"""
    data = request.json
    word = data.get("word", "").lower()
    k = data.get("k", 1000)

    if word not in vocab:
        return jsonify({"error": "word not in vocabulary"}), 400

    similarities = np.dot(vectors, vectors[word_to_idx[word]])
    top_k = np.argsort(similarities)[::-1][:k]
    top_words = [vocab[i] for i in top_k]
    return jsonify({"top_words": top_words})

@app.route("/neighbors", methods=["POST"])
def get_neighbors():
    """returns the neighbors of a word"""
    data = request.json
    word = data.get("word", "").lower()

    if word not in vocab:
        return jsonify({"error": "word not in vocabulary"}), 400

    neighbors = [vocab[i] for i in graph[word_to_idx[word]]]
    return jsonify({"neighbors": neighbors})

@app.route("/healthcheck", methods=["GET"])
def healthcheck():
    return "healthy"

if __name__ == "__main__":
    app.run(debug=True)