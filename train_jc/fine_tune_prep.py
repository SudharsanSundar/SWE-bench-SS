from split_data import *
from voyage import *
from together_model import TogModel
from faiss_index import VoyageEmbedder
from faiss_index import FaissIndex
from faiss_index import *
import json
from transformers import AutoTokenizer
import re
from io_processing import write_jsonl, list_generator

tokenizer = AutoTokenizer.from_pretrained('voyageai/voyage')
tokenizerMix = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-v0.1")


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
            print("DIDNT MAKE CHANGE")
            text_between_issues = full_text
        extracted_issues.append(text_between_issues)

    return extracted_issues


# Same as above, but allows you to just pass in a streamed dataset, and it takes care of the rest
def extract_issue_general(full_text_list):
    extracted_issues = []
    for all_text in full_text_list:
        #print("BEFORE IT WAS", full_text)
        #print("\n and now...")
        # Find the positions of the first and second <issue> tags
        full_text = all_text['text']
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
            print("DIDNT MAKE CHANGE")
            text_between_issues = full_text
        extracted_issues.append(text_between_issues)

    return extracted_issues


# Creates training file formatted for together fting
def generate_jsonl_for_training(index):
    output_file_path = './data_for_fting.jsonl'
    train_set = get_train_set_stream()
    
    with open(output_file_path, 'w') as file:
        lines = []
        train_issues = extract_issue_general(train_set)
        count = 0
        for row, issue in zip(train_set, train_issues):
            # TODO: figure out what the prompt should be
            # TODO: find out how to incorporate this closest issue thing

            # Currently I'm using the most basic prompt, no intelligently finding closest issue etc.
            full_default_prompt = '[INST] ' + row['text'] + ' [/INST]\n\n'
            answer = row['patch']

            # Prep text for jsonl file
            text = full_default_prompt + answer
            json_line = {"text": text}

            if len(tokenizerMix.encode(text)) < 32700:
                lines.append(json_line)
                print('added ex')
                count += 1

        # Save as jsonl file
        write_jsonl(output_file_path, list_generator(lines))

    print(f'Data successfully written to {output_file_path} for num exs', count)


def main():
    # # Get df of these things
    # # test_set, train_set = split_data()  # checked, works good
    # train_set = get_train_set()  # checked, works good
    #
    # # Set up embedding model
    # embedder = VoyageEmbedder()  # looks good
    #
    # # Get all the issues
    # issues = train_set['text']
    # extracted_issues = extract_issue(issues)
    # print("HERE IS AN ISSUE")
    # print(extracted_issues[0])
    #
    # # Create mapping from issue to embedding, TODO: should be embedding to issue?
    # issue_to_embedding = {}
    # i = 0
    #
    # for extracted_issue in extracted_issues:
    #     print(f'This is issue {i}')
    #     issue_to_embedding[extracted_issue] = embedder.create_embedding(extracted_issue)
    #     i += 1
    #
    # # print(issue_to_embedding)

    # Create embedding database
    # embdIndex = FaissIndex(chunks=extracted_issues)
    #
    # embdIndex.build()  ###TODO MAKE SURE NOT RETURNING SAME ONE

    # Massively simplified to get something running. Feel free to build off of it.
    embdIndex = None

    generate_jsonl_for_training(embdIndex)


if __name__ == "__main__":
    main()
