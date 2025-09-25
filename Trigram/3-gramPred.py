import random
from collections import defaultdict
import nltk
from nltk.corpus import brown


# Dataset preparation
def prepare_data():

    all_words = []
    for fileid in brown.fileids():
        words = brown.words(fileid)
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

# Build 3gram model
def build_trigram_model(sentences):
    trigram_counts = defaultdict(int)
    bigram_counts = defaultdict(int)

    for sentence in sentences:
        for i in range(len(sentence) - 2):
            trigram = (sentence[i], sentence[i+1], sentence[i+2])
            trigram_counts[trigram] += 1
            bigram_counts[(sentence[i], sentence[i+1])] += 1

    trigram_probabilities = {}
    for trigram, count in trigram_counts.items():
        context = (trigram[0], trigram[1])
        trigram_probabilities[trigram] = count / bigram_counts[context]

    return trigram_probabilities



def predict_next_word_trigram(trigram_probabilities, context):
    candidates = {tg: prob 
                  for tg, prob in trigram_probabilities.items()
                  if (tg[0], tg[1]) == context}

    if not candidates:
        return "<No prediction>"

    next_word = max(candidates, key=candidates.get)[2]
    return next_word

# Main execution
if __name__ == "__main__":
    sentences = prepare_data()
    trigram_probabilities = build_trigram_model(sentences)
    
while True:
    user_input = input("Enter a phrase (or 'quit' to exit): ")
    if user_input.lower() == "quit":
        break

    tokens = user_input.lower().split()
    if len(tokens) < 2:
        print("Please enter at least two words.")
        continue

    context = (tokens[-2], tokens[-1])  # last two words
    prediction = predict_next_word_trigram(trigram_probabilities, context)
    print("Predicted next word Trigram:", prediction)
