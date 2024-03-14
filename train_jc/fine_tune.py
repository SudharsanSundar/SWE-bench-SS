from split_data import *
from voyage import *
from model import TogModel

def main():  
    test_set, train_set = split_data()

    embedder = VoyageEmbedder()

    # generate the key: issue and value: embedding  
    msg_to_emb = embedder.generate_embeddings(train_set)

    
    model = TogModel()

    for row in train_set:

        gold_patch = "This is an example of what a correct fix looks like: " + embedder.get_closest_patch(row, msg_to_emb)
        # prompt: our issue, code base(?), oracle info, and the relevant gold patch we've just generated

        issue = "Here is the issue to resolve ", row['text']
        #TODO create the prompt 
        # code base = ?
        # oracle info = "Here are files that are relevant to update:"
        # 
        prompt = issue + gold_patch
        
        """
        TODO actually fine tune the model on these prompts
        https://docs.together.ai/docs/fine-tuning-python#start-fine-tuning
        """






if __name__ == "__main__":
    main()