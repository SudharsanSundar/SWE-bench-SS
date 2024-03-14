# import faiss
# import numpy as np
# from openai import OpenAI
# import anthropic

# # TODO: Delete before push!!!


# client = OpenAI(api_key=OAI_API_KEY)
# togClient = OpenAI(api_key=TOG_API_KEY, base_url='https://api.together.xyz')
# antClient = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)


# # """
# # Object that deals with efficiently creating relatively good text embeddings.
# # """
# # class SBertEmbeddingModel:
# #     def __init__(self, model_name="sentence-transformers/multi-qa-mpnet-base-cos-v1"):
# #         self.model = SentenceTransformer(model_name)
# #
# #     def create_embedding(self, text):
# #         return self.model.encode(text)


# """
# Object that creates openai embeddings.
# """
# class OpenAIEmbeddingModel:
#     def __init__(self, model_name="text-embedding-3-small"):
#         self.model_name = model_name

#     def create_embedding(self, text):
#         # Not sure why we need this replace, but this is what boilerplate openai code has
#         embedding_text = text.replace('\n', ' ')
#         return client.embeddings.create(input=[embedding_text], model=self.model_name).data[0].embedding


# """
# Object for creating Voyage embeddings.
# """
# class VoyageEmbedder:
#     def __init__(self, model_name="???"):
#         self.model_name = model_name

#     def create_embedding(self, text):
#         # TODO: add in logic from voyage.py
#         return 


# """
# TODO: fill in
# """
# class FaissIndex:
#     def __init__(self, chunks: list, model=None, k=5):
#         if model is None:
#             self.model = OpenAIEmbeddingModel()
#         else:
#             self.model = model

#         embeddings = []
#         for chunk in chunks:
#             embeddings.append(self.model.create_embedding(chunk))

#         self.embeddings = np.array(embeddings)
#         self.index = None
#         self.k = k

#     def build(self):
#         d = self.embeddings.shape[1]
#         self.index = faiss.IndexFlatIP(d)
#         self.index.add(self.embeddings)

#     def search(self, text):
#         embedding = self.model.create_embedding(text)
#         distances, indices = self.index.search(embedding, k=self.k)
#         return distances, indices


# def cosine_similarity(a, b):
#     return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

