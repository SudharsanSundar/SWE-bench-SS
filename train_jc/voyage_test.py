from faiss_index import VoyageEmbedder, FaissIndex
import numpy as np

textA = "Islamabad is in Pakistan"
textB = "Paris is the capital of France"

model = VoyageEmbedder()
chunks = [textA, textB]
index = FaissIndex(chunks=chunks)
index.build()
result = index.search("What is the capital of Pakistan")
print(result)
#example = result[1]
