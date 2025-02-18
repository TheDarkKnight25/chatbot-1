
# Step 1: Load the Word2Vec Model
import gensim

# Load the pre-trained Word2Vec model
word2vec_model = gensim.models.Word2Vec.load('data/word2vec_model.bin')

# print(word2vec_model.wv.index_to_key[:10])  # Print the first 10 words in the vocabulary

# # Check the model by accessing some word vectors
# print(word2vec_model.wv['and'])  # Just an example to check

# similar_words = word2vec_model.wv.most_similar('offer', topn=10)
# print(similar_words)
# # End of Step 1


# Step 2: Define a Function to Convert Text to Vector
import numpy as np

def text_to_vector(text, model, vector_size=100):
    """
    Convert a sentence into a vector by averaging the word vectors.
    Args:
        text (str): The input sentence.
        model (gensim.models.Word2Vec): The trained Word2Vec model.
        vector_size (int): The dimension of the word vectors.
    Returns:
        np.array: The averaged word vector for the sentence.
    """
    # Ensure that the input text is a string
    if not isinstance(text, str):
        text = str(text)  # Convert to string if not already a string

    words = text.split()
    vectors = []

    for word in words:
        if word in model.wv:
            vectors.append(model.wv[word])

    # If no vectors found, return a zero vector (you can also return a random vector or the mean vector)
    if len(vectors) == 0:
        return np.zeros(vector_size)

    # Compute the average of all word vectors
    sentence_vector = np.mean(vectors, axis=0)
    return sentence_vector
# End of Step 2


# Step 3: Convert Conversation Pairs into Vectors
import pandas as pd

# Load your conversation pairs (already preprocessed)
conversation_pairs_df = pd.read_csv('data/processed_conversation_pairs.csv')
print(conversation_pairs_df.isnull().sum())

conversation_pairs_df['input'] = conversation_pairs_df['input'].fillna('')
conversation_pairs_df['output'] = conversation_pairs_df['output'].fillna('')

# Initialize lists to hold the vectors for inputs and outputs
input_vectors = []
output_vectors = []

# Process each conversation pair
for index, row in conversation_pairs_df.iterrows():
    input_text = row['input']
    output_text = row['output']
    
    if not isinstance(input_text, str):
        print(f"Non-string input: {input_text}")
    if not isinstance(output_text, str):
        print(f"Non-string output: {output_text}")

    # Convert both input and output texts into vectors
    input_vector = text_to_vector(input_text, word2vec_model)
    output_vector = text_to_vector(output_text, word2vec_model)
    
    # Append the vectors to the lists
    input_vectors.append(input_vector)
    output_vectors.append(output_vector)

# Convert the lists into NumPy arrays
input_vectors = np.array(input_vectors)
output_vectors = np.array(output_vectors)

# Check the shapes of the resulting vectors
print(f"Input Vectors Shape: {input_vectors.shape}")
print(f"Output Vectors Shape: {output_vectors.shape}")
# End of Step 3

