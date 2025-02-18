import pandas as pd
import re
import ast
import winsound
from multiprocessing import Pool, cpu_count, Manager
from tqdm import tqdm
import time

def load_data():
    print("Loading movie lines dataset...")
    try:
        lines = pd.read_csv('data/cornell_movie_dialogs_corpus/movie_lines.tsv', 
                    sep='\t', header=None, 
                    names=['lineID', 'characterID', 'movieID', 'character', 'text'], 
                    on_bad_lines='skip')
        print(f"Loaded {len(lines)} lines from the dataset.")
    except Exception as e:
        print(f"Error reading the file: {e}")
        return None, None

    print("First few rows of the dataset:")
    print(lines.head())
    print("Checking for missing values...")
    print(lines.isnull().sum())

    lines = lines.dropna(subset=['lineID', 'characterID', 'movieID', 'character', 'text'])
    print(f"Removing duplicate lines (if any)...")
    before_drop = len(lines)
    lines = lines.drop_duplicates(subset=['lineID'])
    print(f"Dropped {before_drop - len(lines)} duplicate lines.")
    print(f"Dataset shape after cleaning: {lines.shape}")

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
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def update_progress(progress_dict, core_id, total_work):
    """
    Update the progress for a specific core (worker).
    """
    progress_dict[core_id] = progress_dict.get(core_id, 0) + 1
    total_progress = sum(progress_dict.values())
    # Calculate overall progress
    return total_progress / total_work

def create_conversation_pairs_worker(lines, conversations, progress_dict, core_id, max_index='all'):
    """
    Worker function to process each conversation and update progress for the core.
    """
    conversation_pairs = []
    
    # Ensure consistent data types for lineIDs
    lines['lineID'] = lines['lineID'].astype(str).str.strip()  
    conversations['utteranceIDs'] = conversations['utteranceIDs'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

    total_conversations = len(conversations)
    
    if max_index == 'all':
        max_index = total_conversations

    print('Length of conversations:', total_conversations)

    # Loop through each conversation
    for index, row in conversations.iterrows():
        if index >= max_index:
            break
        
        line_ids = row['utteranceIDs']
        
        # Split concatenated lineIDs into individual IDs
        split_line_ids = []
        for line_id in line_ids:
            split_line_ids.extend(re.findall(r'L\d+', line_id))  

        # Create pairs of (input_line, output_line)
        for i in range(len(split_line_ids) - 1):  
            input_line_match = lines[lines['lineID'] == split_line_ids[i]]
            output_line_match = lines[lines['lineID'] == split_line_ids[i + 1]]

            if input_line_match.empty or output_line_match.empty:
                continue  # Skip if no matching lineID

            input_line = input_line_match['text'].values[0]
            output_line = output_line_match['text'].values[0]
            conversation_pairs.append((input_line, output_line))
        
        # Update progress for this core
        progress = update_progress(progress_dict, core_id, max_index)
        print(f"Core {core_id} - Progress: {progress * 100:.2f}%")

    return conversation_pairs

def create_conversation_pairs(lines, conversations, max_index='all'):
    """
    Create conversational pairs using multiprocessing and progress tracker.
    """
    # Create a manager for sharing state between processes
    with Manager() as manager:
        # Shared dictionary to track progress across cores
        progress_dict = manager.dict()

        # Set up a pool of workers (one per core)
        num_cores = cpu_count() - 2
        with Pool(num_cores) as pool:
            # Distribute the task to workers
            results = []
            for i in range(num_cores):
                # Split work among workers
                chunk_size = len(conversations) // num_cores
                start_idx = i * chunk_size
                end_idx = (i + 1) * chunk_size if i != num_cores - 1 else len(conversations)
                worker_conversations = conversations.iloc[start_idx:end_idx]
                
                # Apply worker function
                result = pool.apply_async(create_conversation_pairs_worker, 
                                         (lines, worker_conversations, progress_dict, i, max_index))
                results.append(result)

            # Collect results from all workers
            conversation_pairs = []
            for result in results:
                conversation_pairs.extend(result.get())

        # Convert the conversation pairs to a DataFrame
        conversation_pairs_df = pd.DataFrame(conversation_pairs, columns=['input', 'output'])

    print(f"Total conversation pairs: {len(conversation_pairs_df)}")
    return conversation_pairs_df

def main():
    start_time = time.time()
    print("Starting data preprocessing...")

    lines, conversations = load_data()

    if lines is None or conversations is None:
        print("Error loading data. Exiting preprocessing.")
        return

    print("Preprocessing the 'text' column...")
    lines['text'] = lines['text'].apply(preprocess_text)
    print("Preprocessed data (first few rows):")
    print(lines.head())

    print('Lines shape ', lines.shape)
    print('Conversations shape ', conversations.shape)

    lines.to_csv('data/processed_movie_lines.csv', index=False)
    print("Preprocessed data saved to 'data/processed_movie_lines.csv'.")
    
    print("Creating conversation pairs...")
    conversation_pairs_df = create_conversation_pairs(lines, conversations, max_index=conversations.shape[0])

    conversation_pairs_df.to_csv('data/processed_conversation_pairs.csv', index=False)
    print("Processed conversation pairs saved to 'data/processed_conversation_pairs.csv'.")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time taken to process conversations: {elapsed_time:.2f} seconds.")

if __name__ == "__main__":
    main()
