import json
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from rank_bm25 import BM25Okapi
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
from langchain.embeddings.base import Embeddings


class SentenceTransformerWrapper(Embeddings):
    """Wrapper for SentenceTransformer to match the LangChain embedding interface."""

    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        """Embeds a list of documents."""
        return self.model.encode(texts, convert_to_tensor=False)

    def embed_query(self, text):
        """Embeds a single query."""
        return self.model.encode([text], convert_to_tensor=False)[0]


class RetrievalSystem:
    """A retrieval system combining semantic search with BM25."""

    def __init__(self, data_path,
                 model_name="all-MiniLM-L6-v2",
                 top_k=15,
                 top_n=15,
                 top_r=7,
                 fusion_weight=0.7):
        """
        Initialize the retrieval system.
        
        Args:
            data_path (str): Path to the JSON file containing data.
            model_name (str): Name of the SentenceTransformer model.
            top_k (int): Number of results to retrieve from semantic search.
            top_n (int): Number of results to retrieve from BM25 search.
            top_r (int): Number of final results to return after reranking.
            fusion_weight (float): Weight given to BM25 scores during fusion.
        """
        self.top_k = top_k  # for semantic search
        self.top_n = top_n  # for bm25
        self.top_r = top_r  # top from semantic + bm25
        self.fusion_weight = fusion_weight

        # Load data
        with open(data_path, "r") as f:
            data = json.load(f)

        # Prepare documents and metadata
        self.documents = [{"intend": intend, "description": description} for intend, description in data.items()]
        self.descriptions = [doc["description"] for doc in self.documents]
        self.metadata = [{"intend": doc["intend"]} for doc in self.documents]

        # Initialize embedder and vector store
        self.embedding_model = SentenceTransformerWrapper(model_name)
        self.vector_store = FAISS.from_texts(
            texts=self.descriptions, embedding=self.embedding_model, metadatas=self.metadata
        )

        # Initialize BM25
        tokenized_corpus = [text.split() for text in self.descriptions]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def process_query(self, query, top_r: int=7):
        """
        Process a query to retrieve the top labels using semantic search and BM25.

        Args:
            query (str): The input query.

        Returns:
            list: Top labels after fusion and reranking.
        """
        if top_r != self.top_r:
            self.top_r = top_r

        query_embedding = self.embedding_model.embed_query(query)
        tokenized_query = query.split()

        # Perform semantic search
        semantic_results = self.vector_store.similarity_search_with_score_by_vector(query_embedding, k=self.top_k)

        # Perform BM25 search
        bm25_scores = self.bm25.get_scores(tokenized_query)
        bm25_indices = np.argsort(bm25_scores)[::-1][:self.top_n]
        bm25_results = [(self.documents[idx], bm25_scores[idx]) for idx in bm25_indices]

        # Normalize scores for fusion
        semantic_scores = np.array([score for _, score in semantic_results])
        bm25_scores = np.array([score for _, score in bm25_results])

        scaler = MinMaxScaler()
        normalized_semantic = scaler.fit_transform(semantic_scores.reshape(-1, 1)).flatten()
        normalized_bm25 = scaler.fit_transform(bm25_scores.reshape(-1, 1)).flatten()

        # Weighted fusion
        fused_scores = self.fusion_weight * normalized_bm25 + (1 - self.fusion_weight) * normalized_semantic
        fused_results = [
            {"intend": res[0]["intend"], "description": res[0]["description"], "score": score}
            for res, score in zip(bm25_results, fused_scores)
        ]

        # Reranking
        reranked_results = sorted(fused_results, key=lambda x: x["score"], reverse=True)[:self.top_r]

        # Return top labels
        return [res["intend"] for res in reranked_results]


# exmaple

# data_path = r"draft\labels_with_description.json"
# retrieval_system = RetrievalSystem(data_path, model_name="all-MiniLM-L6-v2")
# query = "can you please repeat my list back to me"
# top_labels = retrieval_system.process_query(query, top_r=10)
# print(top_labels)
