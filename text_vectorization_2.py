import gensim
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import string
import pandas as pd

# Load preprocessed conversation pairs
conversation_pairs_df = pd.read_csv('data/processed_conversation_pairs.csv')

# Tokenize and clean the text data (convert to lowercase, remove punctuation)
def preprocess_for_embedding(text):
    text = text.lower()  # Convert text to lowercase
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    return word_tokenize(text)  # Tokenize text into words

# Prepare the data for Word2Vec (using input and output columns)
corpus = []
for _, row in conversation_pairs_df.iterrows():
    corpus.append(preprocess_for_embedding(row['input']))
    corpus.append(preprocess_for_embedding(row['output']))

# Train the Word2Vec model
model = Word2Vec(corpus, vector_size=100, window=5, min_count=1, workers=4)

# Save the model for future use
model.save('data/word2vec_model.bin')
