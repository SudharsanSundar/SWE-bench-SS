import faiss
import numpy as np
import voyageai
vo = voyageai.Client(api_key="pa-eHDxVlgbEIRYCEU7gmvgjT1FpX2OooksxB2cv2JvXjA")



"""
Object for creating Voyage embeddings.
"""
class VoyageEmbedder:
    def __init__(self, model_name="voyage-code-2"):
        self.model_name = model_name

    # Pass in a row in the dataset, embed the text and return the embedding
    def create_embedding(self, issue):
        result = vo.embed(issue, self.model_name)
        return np.array(result.embeddings[0])


class FaissIndex:
    def __init__(self, chunks: list, model=VoyageEmbedder(), k=5):
  
        self.model = model
        self.chunks = chunks
        embeddings = []
        for chunk in chunks:

            embeddings.append(self.model.create_embedding(chunk))

        self.embeddings = np.array(embeddings)
        self.index = None
        self.k = k

    def build(self):
        print("PRINTING self.embeddings.shape\n")
        #print(self.embeddings.shape)
        d = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(d)
        self.index.add(self.embeddings)

    def search(self, text):
        embedding = self.model.create_embedding(text)
        embedding = np.expand_dims(embedding, 0)
        _, indices = self.index.search(embedding, k=self.k)

        result = [self.chunks[i] for i in indices[0]]
        # make sure it doesn't return the TOP one
        return result[1]

# issues = []
# index = FaissIndex(chunks=issues)
# index.build()
# ###TODO MAKE SURE NOT SAME

# for epoch...
#     best_issue = index.search(issue)