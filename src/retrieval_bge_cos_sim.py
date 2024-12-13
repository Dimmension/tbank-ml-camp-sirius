from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
import numpy as np
import json
import time


class RetrievalSystem_chroma:
    """A retrieval system combining semantic search with BM25."""

    def __init__(self,
                 data_path=r"data\labels_with_description.json",
                 nearest_labels_path=r"data\nearest_labels.json",
                #  embedder_name = "BAAI/bge-large-en-v1.5",
                 embedder_name = "chinchilla04/bge-finetuned-train",
                 reranker_name="BAAI/bge-reranker-v2-m3",
                 top_r=10,
                 top_k=10,
                 threshold_mean_diff=1e-6,
                 threshold_conf_drop=1e-5,
        ):

        self.top_r = top_r  # for semantic search
        self.top_k = top_k  # for BGE reranker
        self.threshold_mean_diff = threshold_mean_diff
        self.threshold_conf_drop = threshold_conf_drop

        # Load data
        with open(data_path, "r") as f:
            data = json.load(f)

        with open(nearest_labels_path, "r") as f:
            self.nearest = json.load(f)

        # Prepare documents and metadata
        self.data = {key: value for key, value in data.items()}
        self.labels = list(self.data.keys())
        self.documents = [{"intend": intend, "description": description} for intend, description in data.items()]
        self.descriptions = [doc["description"] for doc in self.documents]
        self.metadata = [{"intend": doc["intend"]} for doc in self.documents]

        # Initialize HuggingFace BGE embedder
        model_kwargs = {'device': 'cuda'}
        encode_kwargs = {'normalize_embeddings': True}
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=embedder_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )

        # Initialize HuggingFace BGE reranker
        self.reranker = HuggingFaceCrossEncoder(
            model_name=reranker_name,
            model_kwargs=model_kwargs
        )

        # chroma
        self.db = Chroma.from_texts(self.descriptions, self.embedding_model, metadatas=self.metadata)


    def process_query(self, query, top_r, top_k):
        """
        Process a query to retrieve the top labels using EnsembleRetriever.
        """
        if top_r != self.top_r:
            self.top_r = top_r

        if top_k != self.top_k:
            self.top_k = top_k

        results = self.db.similarity_search(query, k=self.top_r)
        label_list = [doc.metadata['intend'] for doc in results]

        # Reranking
        descriptions = [[query, self.get_description(label)] for label in label_list]
        reranking_scores = self.reranker.score(descriptions)
        true_indices = np.argsort(reranking_scores)[::-1]
        top_labels = [label_list[idx] for idx in true_indices[:self.top_k]]

        # Return empty list if query is OOS
        is_oos = self.check_is_oos(reranking_scores[true_indices])
        # is_oos = self.check_is_oos_strict(reranking_scores[true_indices])
        if is_oos:
            return [] #, []

        return top_labels #, reranking_scores[true_indices]

    def get_description(self, label):
        return self.data[label]

    def get_labels(self):
        return self.labels
    
    def get_nearest_labels(self, label):
        return self.nearest[label]

    def check_is_oos(self, values):
        mean_diff = 0
        for i in range(self.top_k - 1):
            mean_diff += values[i] - values[i + 1]

        mean_diff /= (self.top_k - 1)
        if mean_diff <= self.threshold_mean_diff:
            return True
        return False
    

    def check_is_oos_strict(self, values):
        mean_diff = 0
        for i in range(self.top_k - 1):
            mean_diff += values[i] - values[i + 1]

        mean_diff /= (self.top_k - 1)

        top_score = values[0]
        second_score = values[1] if len(values) > 1 else 0
        drop = top_score - second_score

        if mean_diff <= self.threshold_mean_diff or drop < self.threshold_conf_drop:
            return True
        return False



# Example Usage

# retrieval_system = RetrievalSystem()

# s = time.time()
# query = "can you please repeat my list back to me"
# top_labels = retrieval_system.process_query(query, top_r=30, top_m=10)
# e = time.time()
# print(f"retrieval time: {e-s:.3f}")

# print("Top Labels:")
# print(top_labels)
