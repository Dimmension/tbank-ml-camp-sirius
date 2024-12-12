from sentence_transformers import CrossEncoder
from langchain_community.vectorstores import FAISS, ScaNN
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import numpy as np
import json
import time


class RetrievalSystem_small:
    """A retrieval system combining semantic search with BM25."""

    def __init__(self,
                 data_path=r"data\labels_with_description.json",
                #  embedder_name="BAAI/bge-large-en-v1.5",
                 embedder_name = "chinchilla04/bge-finetuned-train",
                 reranker_name="cross-encoder/ms-marco-MiniLM-L-12-v2",
                 top_k=20,
                 threshold=1e-6,
                 fusion_weight=0):
        """
        Initialize the retrieval system.

        Args:
            data_path (str): Path to the JSON file containing labels with descriptions.
            embedder_name (str): Name of the embedding model.
            reranker_name (str): Name of the reranker model.
            top_k (int): Number of results to retrieve from semantic search.
            top_n (int): Number of results to retrieve from BM25 search.
            top_r (int): Number of final results to return after reranking.
            threshold (float): Threshold for checking if query is OOS.
            fusion_weight (float): Weight given to BM25 scores during fusion.
        """
        self.top_k = top_k  # for semantic search
        self.fusion_weight = fusion_weight
        self.threshold = threshold

        # Load data
        with open(data_path, "r") as f:
            data = json.load(f)

        # Prepare documents and metadata
        self.data = {key: value for key, value in data.items()}
        self.labels = list(self.data.keys())
        self.documents = [{"intend": intend, "description": description} for intend, description in data.items()]
        self.descriptions = [doc["description"] for doc in self.documents]
        self.metadata = [{"intend": doc["intend"]} for doc in self.documents]

        # Initialize HuggingFace embedder
        model_kwargs = {'device': 'cuda'}
        encode_kwargs = {'normalize_embeddings': True}
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=embedder_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )

        # chroma
        self.db = Chroma.from_texts(self.descriptions, self.embedding_model, metadatas=self.metadata)

        # Initialize Cross-Encoder reranker
        self.reranker = CrossEncoder(reranker_name, device='cuda')
    
    def process_query(self, query, top_k):
        """
        Process a query to retrieve the top labels using EnsembleRetriever.
        """
        if top_k != self.top_k:
            self.top_k = top_k

        results = self.db.similarity_search(query, k=self.top_k)
        label_list = [doc.metadata['intend'] for doc in results]

        # Reranking
        descriptions = [[query, self.get_description(label)] for label in label_list]
        reranking_scores = self.reranker.predict(descriptions)
        true_indices = np.argsort(reranking_scores)[::-1]
        top_labels = [label_list[idx] for idx in true_indices[:self.top_k]]

        # Return empty list if query is OOS
        is_oos = self.check_is_oos(reranking_scores[true_indices])
        if is_oos:
            return []

        return top_labels

    def get_description(self, label):
        return self.data[label]

    def get_labels(self):
        return self.labels

    def check_is_oos(self, values):
        mean_diff = 0
        for i in range(self.top_k - 1):
            mean_diff += values[i] - values[i + 1]

        mean_diff /= (self.top_k - 1)
        if mean_diff <= self.threshold:
            return True
        return False


# Example Usage

if __name__ == '__main__':
    retrieval_system = RetrievalSystem_small()

    s = time.time()
    query = "can you please repeat my list back to me"
    top_labels = retrieval_system.process_query(query, top_k=10)
    e = time.time()
    print(f"retrieval time: {e-s:.3f}")

    print("Top Labels:")
    print(top_labels)

