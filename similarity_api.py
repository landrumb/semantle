from flask import Flask, request, jsonify
from flask_cors import CORS
from pathlib import Path
from utils import fbin_to_numpy, graph_file_to_list_of_lists, read_vocab
import numpy as np

app = Flask(__name__)
CORS(app, supports_credentials=True, resources={r"/*": {"origins": "*"}})  # enable cross-origin requests

@app.before_request
def handle_preflight():
    if request.method == "OPTIONS":
        response = jsonify({"message": "CORS preflight successful"})
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'POST, GET, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
        return response

MODEL = "word2vec-google-news-300_50000_lowercase"

data_dir = Path("data") / MODEL

print("loading vectors...")
vectors = fbin_to_numpy(data_dir / "base.fbin")
word_to_idx, vocab = read_vocab(data_dir / "vocab.txt")

print("loading graph...")
graph = graph_file_to_list_of_lists(data_dir / "outputs" / "vamana")

@app.route("/get_vocab", methods=["GET"])
def get_vocab():
    """returns the vocabulary list to the client so it can select a target word"""
    return jsonify({"vocab": vocab})

@app.route("/similarity", methods=["POST"])
def get_similarity():
    """returns the similarity between two words"""
    data = request.json
    word1 = data.get("word1", "").lower()
    word2 = data.get("word2", "").lower()

    if word1 not in vocab or word2 not in vocab:
        return jsonify({"error": "one or both words not in vocabulary"}), 400

    similarity = float(np.dot(vectors[word_to_idx[word1]], vectors[word_to_idx[word2]]))
    return jsonify({"similarity": similarity})

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

if __name__ == "__main__":
    app.run(debug=True)