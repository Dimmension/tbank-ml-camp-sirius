# import json
# import numpy as np
# from langchain_chroma import Chroma

# from langchain_community.embeddings import HuggingFaceBgeEmbeddings
# from langchain_community.cross_encoders import HuggingFaceCrossEncoder
# from langchain.retrievers import EnsembleRetriever
# from langchain_community.retrievers import BM25Retriever


# class RetrievalSystem:
#     """A retrieval system combining semantic search with BM25."""

#     def __init__(self,
#                  data_path="././data/labels_with_description.json",
#                  nearest_labels_path="././data/nearest_labels.json", 
#                  model_name="chinchilla04/bge-finetuned-train",
#                  reranker_name="BAAI/bge-reranker-v2-m3",
#                  top_k=60,
#                  top_n=60,
#                  top_r=40,
#                  top_m=30,
#                  theshhold=1e-6,
#                  fusion_weight=0.3,
#         ):
#         """
#         Initialize the retrieval system.

#         Args:
#             data_path (str): Path to the JSON file containing labels with descriptions.
#             model_name (str): Name of the BGE model.
#             reranker_name (str): Name of the BGE reranker.
#             top_k (int): Number of results to retrieve from semantic search.
#             top_n (int): Number of results to retrieve from BM25 search.
#             top_r (int): Number of final results to return after reranking.
#             theshhold (int): Threshold for checking if query is OOS.
#             fusion_weight (float): Weight given to BM25 scores during fusion.
#             query_instruction (str): Instruction for query embedding.
#         """
#         self.top_k = top_k  # for semantic search
#         self.top_n = top_n  # for BM25
#         self.top_r = top_r  # top from semantic + BM25
#         self.top_m = top_m  # for BGE reranking
#         self.fusion_weight = fusion_weight
#         self.theshhold = theshhold

#         # Load data
#         with open(data_path, "r") as f:
#             data = json.load(f)
            
#         with open(nearest_labels_path, "r") as f:
#             self.nearest = json.load(f)

#         # Prepare documents and metadata
#         self.data = {key: value for key, value in data.items()}
#         self.labels = list(self.data.keys())
#         self.documents = [{"intend": intend, "description": description} for intend, description in data.items()]
#         self.descriptions = [doc["description"] for doc in self.documents]
#         self.metadata = [{"intend": doc["intend"]} for doc in self.documents]

#         # Initialize HuggingFace BGE embedder
#         model_kwargs = {'device': 'cuda'}
#         encode_kwargs = {'normalize_embeddings': True}
#         self.embedding_model = HuggingFaceBgeEmbeddings(
#             model_name=model_name,
#             model_kwargs=model_kwargs,
#             encode_kwargs=encode_kwargs,
#         )

#         # Initialize HuggingFace BGE reranker
#         self.reranker = HuggingFaceCrossEncoder(
#             model_name=reranker_name,
#             model_kwargs=model_kwargs
#         )
        
#         self.vector_store = Chroma.from_texts(
#             texts=self.descriptions, embedding=self.embedding_model, metadatas=self.metadata
#         )
        
#         # Initialize BM25 retriever
#         self.bm25_retriever = BM25Retriever.from_texts(self.descriptions, metadatas=self.metadata)

#         # Initialize ensemble retriever
#         self.ensemble_retriever = EnsembleRetriever(
#             retrievers=[
#                 self.vector_store.as_retriever(search_type="similarity", search_kwargs={"k": self.top_k}),
#                 self.bm25_retriever,
#             ],
#             weights=[1 - fusion_weight, fusion_weight]
#         )
    
#     def process_query(self, query, top_r, top_m):
#         """
#         Process a query to retrieve the top labels using EnsembleRetriever.
#         """
#         self.top_r, self.top_m = top_r, top_m
#         # Perform ensemble retrieval
#         retrieval_results = self.ensemble_retriever.invoke(query, c=100)

#         # Extract top results
#         results_top_r = retrieval_results[:top_r]
#         label_list = [res.metadata["intend"] for res in results_top_r]

#         # Reranking
#         descriptions = [[query, self.get_description(label)] for label in label_list]
#         reranking_scores = self.reranker.score(descriptions)
#         true_indices = np.argsort(reranking_scores)[::-1]
#         top_labels = [label_list[idx] for idx in true_indices[:top_m]]

#         # Return empty list if query is OOS
#         is_oos = self.check_is_oos(reranking_scores[true_indices])
#         if is_oos:
#             return []

#         return top_labels
    
    
#     def get_description(self, label):
#         return self.data[label]

#     def get_labels(self):
#         return self.labels
    
#     def get_nearest_labels(self, label):
#         return self.nearest[label]

#     def check_is_oos(self, values):
#         mean_diff = 0
#         for i in range(self.top_m - 1):
#             mean_diff += values[i] - values[i + 1]

#         mean_diff /= (self.top_m - 1)
#         if mean_diff <= self.theshhold:
#             return True
#         return False
    
    
    
from langchain_community.vectorstores import FAISS
from langchain_chroma import Chroma
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
import numpy as np
import json
import time


class RetrievalSystem:
    """A retrieval system combining semantic search with BM25."""

    def __init__(self,
                 data_path="././data/labels_with_description.json",
                 nearest_labels_path="././data/nearest_labels.json",
                #  embedder_name = "BAAI/bge-large-en-v1.5",
                 embedder_name = "chinchilla04/bge-finetuned-train",
                 reranker_name="BAAI/bge-reranker-v2-m3",
                 top_r=20,
                 top_k=10,
                 threshold_mean_diff=1e-6,
                 threshold_conf_drop=1e-5,
                 fusion_weight=0.3,
        ):
        """
        Initialize the retrieval system.

        Args:
            data_path (str): Path to the JSON file containing labels with descriptions.
            model_name (str): Name of the BGE model.
            reranker_name (str): Name of the BGE reranker.
            top_k (int): Number of results to retrieve from semantic search.
            top_r (int): Number of final results to return after reranking.
            theshhold (int): Threshold for checking if query is OOS.
            fusion_weight (float): Weight given to BM25 scores during fusion.
            query_instruction (str): Instruction for query embedding.
        """
        self.top_r = top_r  # top from FAISS + BM25
        self.top_k = top_k  # for BGE reranking
        self.fusion_weight = fusion_weight
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

        # Create FAISS vector store
        self.vector_store = FAISS.from_texts(
            texts=self.descriptions, embedding=self.embedding_model, metadatas=self.metadata
        )
        
        # chroma
        # self.vector_store = Chroma.from_texts(
        #     texts=self.descriptions, embedding=self.embedding_model, metadatas=self.metadata
        # )

        # Initialize BM25 retriever
        self.bm25_retriever = BM25Retriever.from_texts(self.descriptions, metadatas=self.metadata)

        # Initialize ensemble retriever
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[
                self.vector_store.as_retriever(search_type="similarity", search_kwargs={"k": self.top_r}),
                self.bm25_retriever,
            ],
            weights=[1 - fusion_weight, fusion_weight]
        )

    def process_query(self, query, top_r, top_k):
        """
        Process a query to retrieve the top labels using EnsembleRetriever.
        """
        if top_r != self.top_r:
            self.top_r = top_r
            
        if top_k != self.top_k:
            self.top_k = top_k

        # Perform ensemble retrieval
        retrieval_results = self.ensemble_retriever.invoke(query, c=100)
        results_top_r = retrieval_results[:self.top_r]
        label_list = [res.metadata["intend"] for res in results_top_r]

        # Reranking
        descriptions = [[query, self.get_description(label)] for label in label_list]
        reranking_scores = self.reranker.score(descriptions)
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
# top_labels = retrieval_system.process_query(query, top_r=20, top_k=10)
# e = time.time()
# print(f"retrieval time: {e-s:.3f}")

# print("Top Labels:")
# print(top_labels)