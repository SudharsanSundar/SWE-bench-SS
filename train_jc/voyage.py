# ### GET THE VOYAGE STUFF

# import voyageai
# from split_data import get_train_set
# vo = voyageai.Client(api_key="YOUR_API_KEY")
# import numpy as np

# class VoyageEmbedder:
#     def __init__(self, model_name="voyage-code-2"):
#         self.model_name = model_name

#     # Pass in a row in the dataset, embed the text and return the embedding
#     def embed_one_item(self, row):
#         embedding_input = row['text']
#         result = vo.embed(embedding_input, self.model_name)
#         return result.embeddings

#     def generate_embeddings(self, data):
#         # key: commit message  | value: its embedding
#         msg_to_emb = {}
#         for row in data:
#             print(f"Here is the text { row['text'] }")
#             msg_to_emb[row['text']] = self.embed_one_item(row)

#         return msg_to_emb
    
    
#     """
#     Pass in a row in the dataset. Pass in the dictionary of issues ("text") mapped to embeddings. 
#     Based on the issue in the provided row (desired_row), find the issue that is most similar in cosine similarity,
#     and return that issue's golden patch. 
#     """
#     def get_closest_patch(self, desired_row, msg_to_emb):
#         greatest_similarity = float('-inf') 
#         golden_patch = ""
#         for row in msg_to_emb:
#             if desired_row['text'] != row['text']:
#                 desired_row_embedding = self.embed_one_item(desired_row)
#                 cosine_similarity = cosine_similarity(desired_row_embedding, msg_to_emb[row])
#                 greatest_similarity = max(greatest_similarity, cosine_similarity)
#                 golden_patch = row['patch']
#         return golden_patch



# def cosine_similarity(a, b):
#     return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))