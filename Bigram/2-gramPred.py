import random
from collections import defaultdict
import nltk
from nltk.corpus import gutenberg


# Dataset preparation
def prepare_data():
    # Load text from all Gutenberg books
    all_words = []
    for fileid in gutenberg.fileids():
        words = gutenberg.words(fileid)
        all_words.extend(words)
    
    sentences = []
    current_sentence = []
    
    for word in all_words:
        if word in [".", "!", "?"]:  # sentence boundary
            if current_sentence:
                sentences.append(current_sentence)
                current_sentence = []
        else:
            current_sentence.append(word.lower())
    return sentences

# Build the bigram model
def build_bigram_model(sentences):
    bigram_counts = defaultdict(int)
    unigram_counts = defaultdict(int)
    
    for sentence in sentences:
        for i in range(len(sentence) - 1):
            bigram = (sentence[i], sentence[i + 1])
            bigram_counts[bigram] += 1
            unigram_counts[sentence[i]] += 1
        if sentence:  # Check if sentence is not empty
            unigram_counts[sentence[-1]] += 1  # last word
    
    bigram_probabilities = {}
    for bigram, count in bigram_counts.items():
        bigram_probabilities[bigram] = count / unigram_counts[bigram[0]]
    
    return bigram_probabilities

# Predict the next word
def predict_next_word(bigram_probabilities, input_sentence):
    words = input_sentence.lower().split()
    last_word = words[-1]
    
    # Filter bigrams starting with the last word
    candidates = {bigram: prob 
                  for bigram, prob in bigram_probabilities.items() 
                  if bigram[0] == last_word}
    
    if not candidates:
        return "<No prediction>"
    
    # Pick the word with the highest probability
    next_word = max(candidates, key=candidates.get)[1]
    return next_word




# Main execution
if __name__ == "__main__":
    sentences = prepare_data()
    bigram_probabilities = build_bigram_model(sentences)
    
    while True:
        user_input = input("Enter a phrase (or 'quit' to exit): ")
        if user_input.lower() == "quit":
            break
        prediction = predict_next_word(bigram_probabilities, user_input)
        print("Predicted next word Bigram:", prediction)