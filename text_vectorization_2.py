import gensim
from gensim.models import Word2Vec
import string
import re
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt_tab')

# Load preprocessed conversation pairs
conversation_pairs_df = pd.read_csv('data/processed_conversation_pairs.csv')

# Tokenize and clean the text data (convert to lowercase, remove punctuation)
def preprocess_for_embedding(text):
    """
    Preprocess the text for embedding by ensuring it's a string and then applying cleaning steps.
    """
    if isinstance(text, str):  # Ensure it's a string
        text = text.lower()  # Convert text to lowercase
        text = re.sub(r'[^a-z\s]', '', text)  # Remove non-alphabetic characters
        text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
        # Tokenize text into words
        tokens = word_tokenize(text)  # Use word_tokenize to get words
    else:
        tokens = []  # Return empty list if it's not a string
    
    return tokens

# Example to check tokenization
example_text = "no doubt sir but i am endowed with talent and you with money"
print(preprocess_for_embedding(example_text))  # This should print a list of words


# Prepare the data for Word2Vec (using input and output columns)
corpus = []
for _, row in conversation_pairs_df.iterrows():
    corpus.append(preprocess_for_embedding(row['input']))
    corpus.append(preprocess_for_embedding(row['output']))

# Train the Word2Vec model
model = Word2Vec(corpus, vector_size=100, window=5, min_count=1, workers=4, epochs=20)

# Check if the word 'hello' is in the vocabulary
if 'offer' in model.wv:
    print("Word 'offer' found in vocabulary.")
else:
    print("Word 'offer' NOT found in vocabulary.")


# Save the model for future use
model.save('data/word2vec_model.bin')

print("Word2Vec model saved successfully.")
