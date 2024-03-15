from split_data import *
from voyage import *
from together_model import TogModel
from faiss_index import VoyageEmbedder
from faiss_index import FaissIndex
from faiss_index import *
import json
from transformers import AutoTokenizer
import re

tokenizer = AutoTokenizer.from_pretrained('voyageai/voyage')

def extract_issue(full_text_list):
    extracted_issues = []
    for full_text in full_text_list:
        #print("BEFORE IT WAS", full_text)
        #print("\n and now...")
        # Find the positions of the first and second <issue> tags
        first_issue_pos = full_text.find("<issue>") + len("<issue>")
        #print("INDEX OF FIRST_ISSUE_POS", first_issue_pos)
        second_issue_pos = full_text.find("</issue>", first_issue_pos)
        #print("INDEX OF SECOND_ISSUE_POS", second_issue_pos)
        text_between_issues = ""
# Extract the text between the first and second <issue> tags
        if first_issue_pos != -1 and second_issue_pos != -1:
            text_between_issues = full_text[first_issue_pos:second_issue_pos].strip()
         #   print("MADE CHANGE")
          #  print(text_between_issues)
        else:
           # print("DIDNT MAKE CHANGE")
            text_between_issues = full_text
        extracted_issues.append(text_between_issues)
        #print("------")
            
    return extracted_issues

def main(): 

    test_set, train_set = split_data()

    embedder = VoyageEmbedder()
    # get all the issues TODO FORMAT CORRECTLY
    issues = train_set['text']
    extracted_issues = extract_issue(issues)
    print("HERE IS AN ISSUE")
    print(extracted_issues[0])
    #print("HERE IS ONE EXAMPLE FOR TOKENIZE COUNTING")
    #print("this is one item", issues[0])
    issue_to_embedding = {}
    i = 0
    
    #num_tokens = len(vo.tokenize(issues))
    #print("this is the number of tokens", num_tokens)
     
    for extracted_issue in extracted_issues:    
        print(f'This is issue {i}')
        i += 1
        #print(extracted_issue)
        issue_to_embedding[extracted_issue] = embedder.create_embedding(extracted_issue)
    print(issue_to_embedding)

    index = FaissIndex(chunks=extracted_issues)
    #print("this is index", index)
    index.build()       ###TODO MAKE SURE NOT RETURNING SAME ONE
    generate_jsonl_for_training(index)


def generate_jsonl_for_training(index):
    output_file_path = '/Users/Jessica/Downloads/SWE-bench-SS/train_jc/data_for_training_jc.jsonl'
    test_set, train_set = split_data()
    
    with open(output_file_path, 'w') as file:
        for row in train_set:  
            # prompt: our issue, code base(?), oracle info, and the relevant gold patch we've just generated

            issue = "Here is the issue to resolve " + row['text']
            
            best_issue = index.search(issue)
            
            example_gold_patch = "This is an example of what a correct fix looks like: " + best_issue
            #TODO create the prompt 
            # code base = ?
            # oracle info = "Here are files that are relevant to update:"
            # 
            
            input = issue + example_gold_patch
            print("this is input", input)
            actual_gold_patch = row['patch']
            text = """<s>[INST] {input} {example_gold_patch} [/INST] {actual_gold_patch} </s>""" 
            json_object = {"text": text}
            # Convert the dictionary to a JSON string
            json_string = json.dumps(json_object)
            # Write the JSON string to the file, followed by a newline character
            file.write(json_string + '\n')
    
    print(f'Data successfully written to {output_file_path}')



if __name__ == "__main__":
    main()