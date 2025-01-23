import gensim.downloader as api
import random
import os

class SemantleGame:
    def __init__(self, model=None):
        """
        Initializes the Semantle game.
        Loads a word embedding model and selects a random target word.
        """
        if model is None:
            print("Loading word embedding model...")
            self.model = api.load('glove-wiki-gigaword-50')  # You can choose a different model if you like
        else:
            self.model = model
        self.vocab = list(self.model.key_to_index.keys())
        self.target_word = random.choice(self.vocab)
        self.guesses = []
        self.solved = False
        self._compute_target_similarities()
        print("Target word selected. Let's start the game!")

    def _compute_target_similarities(self):
        """
        Computes the similarities between the target word and all other words in the vocabulary.
        """
        print("Computing similarities for all vocabulary words...")
        self.target_similarities = {}
        for word in self.vocab:
            if word == self.target_word:
                continue
            sim = self.model.similarity(self.target_word, word)
            self.target_similarities[word] = sim
        # Sort words by similarity
        self.sorted_words = sorted(self.target_similarities.items(), key=lambda x: x[1], reverse=True)
        # Map from word to rank
        self.word_ranks = {word: rank+1 for rank, (word, sim) in enumerate(self.sorted_words)}
        print("Similarities computed.")

    def make_guess(self, guess_word):
        """
        Processes a player's guess, updating the game state and providing feedback.
        """
        guess_word = guess_word.lower()
        if self.solved:
            print("You've already guessed the word!")
            return
        if guess_word not in self.model.key_to_index:
            print(f"'{guess_word}' is not in the vocabulary.")
            return
        sim = self.model.similarity(self.target_word, guess_word)
        rank = self.get_rank(guess_word)
        self.guesses.append({'word': guess_word, 'similarity': sim, 'rank': rank})
        if guess_word == self.target_word:
            self.solved = True
            print(f"Congratulations! You've guessed the word '{self.target_word}'!")
        else:
            print(f"Guess: {guess_word}")
            print(f"Similarity: {sim:.4f}")
            if rank:
                print(f"Rank: {rank}")
            else:
                print("Rank: > 1000")

    def get_rank(self, word):
        """
        Retrieves the rank of a word based on its similarity to the target word.
        """
        return self.word_ranks.get(word, None)

    def display_best_guesses(self, n=10):
        """
        Displays the top n guesses made so far.
        """
        print("\nBest guesses so far:")
        sorted_guesses = sorted(self.guesses, key=lambda x: x['similarity'], reverse=True)
        for i, guess in enumerate(sorted_guesses[:n]):
            rank = guess['rank'] if guess['rank'] else "> 1000"
            print(f"{i+1}. {guess['word']} - Similarity: {guess['similarity']:.4f}, Rank: {rank}")

    def __str__(self):
        """
        Provides an informative string representation of the game state.
        """
        if self.solved:
            return f"Game solved! The word was '{self.target_word}'. Total guesses: {len(self.guesses)}."
        else:
            return f"Semantle Game - {len(self.guesses)} guesses made. Keep trying!"

    def play(self):
        """
        Allows the game to be played from the command line.
        """
        while not self.solved:
            guess_word = input("Enter your guess: ").strip()
            os.system('cls' if os.name == 'nt' else 'clear')
            self.make_guess(guess_word)
            self.display_best_guesses()

    def give_hint(self):
        """
        Provides a hint by revealing the first letter of the target word.
        """
        print(f"Hint: The target word starts with '{self.target_word[0]}'.")

# Command-line interface to play the game
if __name__ == "__main__":
    game = SemantleGame()
    game.play()
