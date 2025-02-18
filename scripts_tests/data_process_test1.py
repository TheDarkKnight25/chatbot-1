import pandas as pd
import re
import ast  # Import ast to safely evaluate the string list into an actual list
import winsound  # For beep sound (Windows)
from multiprocessing import Pool, cpu_count, Manager
from tqdm import tqdm
import time

def load_data():
    """
    Load the Cornell movie dialogs dataset and handle potential issues like 
    mismatched row lengths or unexpected data formatting.
    """
    print("Loading movie lines dataset...")

    try:
        # Try reading the data with error handling
        # Load the data with TSV (tab-separated) format
        lines = pd.read_csv('data/cornell_movie_dialogs_corpus/movie_lines.tsv', 
                    sep='\t', header=None, 
                    names=['lineID', 'characterID', 'movieID', 'character', 'text'], 
                    on_bad_lines='skip')  # Skip any bad lines
        print(f"Loaded {len(lines)} lines from the dataset.")
    except Exception as e:
        print(f"Error reading the file: {e}")
        return None, None

    # Check the first few lines to understand the data structure
    print("First few rows of the dataset:")
    print(lines.head())

    # Ensure there are no null or empty values in the key columns
    print("Checking for missing values...")
    print(lines.isnull().sum())  # Print the count of null values for each column

    # Drop rows with missing values (if any)
    lines = lines.dropna(subset=['lineID', 'characterID', 'movieID', 'character', 'text'])

    # Check for duplicate lines and remove them
    print(f"Removing duplicate lines (if any)...")
    before_drop = len(lines)
    lines = lines.drop_duplicates(subset=['lineID'])
    print(f"Dropped {before_drop - len(lines)} duplicate lines.")

    # Additional debug print to inspect the dataset's shape
    print(f"Dataset shape after cleaning: {lines.shape}")

    # Load the conversations data (for this example, it's being skipped for brevity)
    try:
        conversations = pd.read_csv('data/cornell_movie_dialogs_corpus/movie_conversations.tsv', 
                                    sep='\t', header=None, 
                                    names=['character1', 'character2', 'movieID', 'utteranceIDs'])
        print(f"Loaded {len(conversations)} conversations.")
    except Exception as e:
        print(f"Error reading the conversations file: {e}")
        return lines, None

    return lines, conversations


def preprocess_text(text):
    """
    Preprocess the text by converting to lowercase, removing special characters,
    and cleaning up unnecessary whitespace.
    """
    # Convert text to lowercase
    text = text.lower()

    # Remove special characters, punctuation, and numbers (keeping only alphabets and spaces)
    text = re.sub(r'[^a-z\s]', '', text)

    # Remove extra spaces (leading, trailing, and extra spaces between words)
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def create_conversation_pairs(lines, conversations, max_index='all'):
    """
    Create conversational pairs (input, output) from the movie conversations.
    Each pair consists of two consecutive lines from a conversation.
    """
    conversation_pairs = []
    
    # Ensure consistent data types for lineIDs
    lines['lineID'] = lines['lineID'].astype(str).str.strip()  
    conversations['utteranceIDs'] = conversations['utteranceIDs'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

    print("First few lineIDs from movie_lines:")
    print(lines['lineID'].head())

    # Check a few random conversation IDs and their associated lineIDs
    print("Sample conversation entries:")
    for i in range(5):  # Inspect the first 5 conversations
        print(f"Conversation {i}: {conversations.iloc[i]['utteranceIDs']}")

    total_conversations = len(conversations)
    
    # If max_index is 'all', use the total number of conversations
    if max_index == 'all':
        max_index = total_conversations

    # Loop through each conversation
    for index, row in conversations.iterrows():
        if index >= max_index:  # Stop if max_index is reached
            break
        
        line_ids = row['utteranceIDs']
        
        # Ensure that each line_id is a separate ID if they are concatenated
        split_line_ids = []
        for line_id in line_ids:
            # Split concatenated lineIDs (e.g., 'L666485L666486') into individual IDs
            split_line_ids.extend(re.findall(r'L\d+', line_id))  

        print(f"Conversation {index} split line_ids: {split_line_ids}")
        
        # Create pairs of (input_line, output_line) from the conversation
        for i in range(len(split_line_ids) - 1):  
            input_line_match = lines[lines['lineID'] == split_line_ids[i]]
            output_line_match = lines[lines['lineID'] == split_line_ids[i + 1]]

            if input_line_match.empty or output_line_match.empty:
                continue  # Skip if no matching lineID for input or output

            input_line = input_line_match['text'].values[0]
            output_line = output_line_match['text'].values[0]
            conversation_pairs.append((input_line, output_line))

        # Show progress as percentage
        progress = (index + 1) / max_index * 100
        print(f"Processing conversation {index + 1}/{max_index} - {progress:.2f}%")

    # Play a beep sound (Windows-specific)
    winsound.Beep(1000, 500)  # Beep sound at 1000 Hz for 500 ms

    # Convert the conversation pairs to a DataFrame
    conversation_pairs_df = pd.DataFrame(conversation_pairs, columns=['input', 'output'])

    print(f"Total conversation pairs: {len(conversation_pairs_df)}")
    return conversation_pairs_df




def main():
    """
    The main function to load the data, preprocess it, and save the cleaned dataset.
    """
    start_time = time.time()
    print("Starting data preprocessing...")

    # Load the data
    lines, conversations = load_data()

    if lines is None or conversations is None:
        print("Error loading data. Exiting preprocessing.")
        return

    # Apply text preprocessing to the 'text' column
    print("Preprocessing the 'text' column...")
    lines['text'] = lines['text'].apply(preprocess_text)

    # Print the first few rows after preprocessing to verify
    print("Preprocessed data (first few rows):")
    print(lines.head())

    # Save the preprocessed data for future use
    lines.to_csv('data/processed_movie_lines.csv', index=False)
    print("Preprocessed data saved to 'data/processed_movie_lines.csv'.")

    # Create conversation pairs
    print("Creating conversation pairs...")
    conversation_pairs_df = create_conversation_pairs(lines, conversations, max_index=20000)

    # Save the conversation pairs
    conversation_pairs_df.to_csv('data/processed_conversation_pairs.csv', index=False)
    print("Processed conversation pairs saved to 'data/processed_conversation_pairs.csv'.")

    # Calculate the time taken
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time taken to process conversations: {elapsed_time:.2f} seconds.")


if __name__ == "__main__":
    main()
