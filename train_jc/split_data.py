### WE ARE GOING TO GET THE TRAIN AND TEST DATA SPLIT

import os
from datasets import load_dataset
import pandas as pd



"""
We were able to successfully evaluate on ~600 repos/patches. 
This function returns a list of which repos/patches those are. 
"""
def get_test_names(directory_path):
    
    # Initialize the list to hold the processed file names
    correctly_evaled = []

    # Iterate through each file in the directory
    for filename in os.listdir(directory_path):
        # Check if the file name follows the expected pattern
        if filename.endswith('.gpt-3.5-turbo-1106.eval.log'):
            # Remove the ".gpt-3.5-turbo-1106.eval.log" part from the file name
            clean_name = filename.replace('.gpt-3.5-turbo-1106.eval.log', '')
            # Add the cleaned name to the "test set" list
            correctly_evaled.append(clean_name)

    # Print the "test set" list to verify the results
    #print("this is the correctly evaled set", correctly_evaled)
    #print(len(correctly_evaled))
    return correctly_evaled

"""
This function gets 200 random examples from the examples that we 
were able to successfully evaluate on. This is our test set that we 
will evaluate our methods on. 
"""
def get_test_set(correctly_evaled):
    # get the data that we were able to correctly eval, so we can use this for evaluation
    dataset = load_dataset("princeton-nlp/SWE-bench_oracle")
   
    train_data = dataset.filter(lambda example: example['instance_id'] in correctly_evaled)['test']
    
    shuffled_data = train_data.shuffle(seed=42)
    # Select the first 200 elements
    random_subset = shuffled_data.select(range(200))   
    test_df = pd.DataFrame(random_subset)
    print(test_df)
    return random_subset
    
"""
This functions gets 800 examples randomly from everything that has NOT 
correctly evaluated on. This is our training data.
"""
def get_train_set():
    # get the data that we were able to correctly eval, so we can use this for evaluation
    dataset = load_dataset("princeton-nlp/SWE-bench_oracle")
    
    #train_set = dataset.filter(lambda example: example['instance_id'] not in correctly_evaled)

    #shuffled_data = train_set['train'].shuffle(seed=42)
    #train_df = pd.DataFrame(shuffled_data)
    # Select the first 200 elements
    #random_subset = shuffled_data.select(range(800))
    #train_df = pd.DataFrame(random_subset)

    # gets the train examples from this dataset
    shuffled_data = dataset['train'].shuffle(seed=42)

    # Select the first 800 elements
    random_subset = shuffled_data.select(range(800))
    return random_subset

"""
Here we return the train and test data
using the directory path. 
"""
def split_data():
    # Directory containing the files
    directory_path = '/Users/Jessica/Downloads/SWE-bench-SS/outputs/gpt-3.5-turbo-1106'
    correctly_evaled = get_test_names(directory_path)
    test_set = get_test_set(correctly_evaled)
    train_set = get_train_set()
    return train_set, test_set


# def generate_jsonl_for_trainingtraining(train_set):
#     for row in train_set:
#         issue = "Here is the issue to resolve ", row['text']
#         example_gold_patch = "This is an example of what a correct fix looks like: " + embedder.get_closest_patch(row, msg_to_emb)
#         prompt = """<s>[INST] {issue} {example_gold_patch} [/INST] {actual_gold_patch} </s>""" 

#     issue = "Here is the issue to resolve ", row['text']

#     example_gold_patch = "This is an example of what a correct fix looks like: " + embedder.get_closest_patch(row, msg_to_emb)
#     #TODO create the prompt 
#     # code base = ?
#     # oracle info = "Here are files that are relevant to update:"
#     # 
#     prompt = issue + gold_patch
        
#     output_file_path = '/Users/Jessica/Downloads/SWE-bench-SS/train_jc'

#     with open(output_file_path, 'w') as file:
#         for text in text_values:
#             # Create a dictionary for the current text value
#             json_object = {"text": text}
#             # Convert the dictionary to a JSON string
#             json_string = json.dumps(json_object)
#             # Write the JSON string to the file, followed by a newline character
#             file.write(json_string + '\n')

#     print(f'Data successfully written to {output_file_path}')