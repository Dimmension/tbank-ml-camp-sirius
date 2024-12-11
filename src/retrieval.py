import json
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from rank_bm25 import BM25Okapi
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
import spacy


class RetrievalSystem:
    """A retrieval system combining semantic search with BM25."""

    def __init__(self,
                 data_path=r"data\labels_with_description.json",
                #  model_name="BAAI/bge-large-en",
                 model_name = "BAAI/bge-large-en-v1.5",
                 reranker_name="BAAI/bge-reranker-v2-m3",
                 top_k=60,
                 top_n=60,
                 top_r=30,
                 top_m=30,
                 theshhold=1e-6,
                 fusion_weight=0.7,
                #  query_instruction="retrieve most appropriate labels: ",
        ):
        """
        Initialize the retrieval system.

        Args:
            data_path (str): Path to the JSON file containing labels with descriptions.
            model_name (str): Name of the BGE model.
            reranker_name (str): Name of the BGE reranker.
            top_k (int): Number of results to retrieve from semantic search.
            top_n (int): Number of results to retrieve from BM25 search.
            top_r (int): Number of final results to return after reranking.
            theshhold (int): Threshold for checking if query is OOS.
            fusion_weight (float): Weight given to BM25 scores during fusion.
            query_instruction (str): Instruction for query embedding.
        """
        self.top_k = top_k  # for semantic search
        self.top_n = top_n  # for BM25
        self.top_r = top_r  # top from semantic + BM25
        self.top_m = top_m  # for BGE reranking
        self.fusion_weight = fusion_weight
        self.theshhold = theshhold

        # Load data
        with open(data_path, "r") as f:
            data = json.load(f)

        # Prepare documents and metadata
        self.data = {key: value for key, value in data.items()}
        self.labels = list(self.data.keys())
        self.documents = [{"intend": intend, "description": description} for intend, description in data.items()]
        self.descriptions = [doc["description"] for doc in self.documents]
        self.metadata = [{"intend": doc["intend"]} for doc in self.documents]

        # Initialize HuggingFace BGE embedder
        model_kwargs = {'device': 'cuda'}
        encode_kwargs = {'normalize_embeddings': True}
        self.embedding_model = HuggingFaceBgeEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
            # query_instruction=query_instruction
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

        # Initialize spaCy for advanced BM25 preprocessing
        self.nlp = spacy.load("en_core_web_sm")

        # Tokenize corpus using spaCy
        tokenized_corpus = [self._tokenize_and_preprocess(text) for text in self.descriptions]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def _tokenize_and_preprocess(self, text):
        """
        Tokenize and preprocess text using spaCy (lemmatization and stopword removal).
        """
        doc = self.nlp(text)
        tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
        return tokens

    def process_query(self, query, top_r: int = 7, top_m: int = 5):
        """
        Process a query to retrieve the top labels using semantic search and BM25.
        """
        if top_r != self.top_r:
            self.top_r = top_r

        if top_m != self.top_m:
            self.top_m = top_m

        # Generate query embedding
        query_embedding = self.embedding_model.embed_query(query)
        tokenized_query = self._tokenize_and_preprocess(query)

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

        # Get top results
        results_top_r = fused_results[:self.top_r]
        label_list = np.array([res["intend"] for res in results_top_r])

        # Reranking
        descriptions = [[query, self.get_description(label)] for label in label_list]
        reranking_scores = self.reranker.score(descriptions)
        true_indices = np.argsort(reranking_scores)[::-1]
        top_labels = label_list[true_indices][:self.top_m].tolist()

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
        for i in range(self.top_r - 1):
            mean_diff += values[i] - values[i + 1]

        mean_diff /= (self.top_r - 1)
        if mean_diff <= self.theshhold:
            return True
        return False



# Example Usage

# data_path = r"data\labels_with_description.json"
# retrieval_system = RetrievalSystem(data_path)

# s = time.time()
# query = "can you please repeat my list back to me"
# top_labels = retrieval_system.process_query(query, top_r=10)
# e = time.time()
# print(f"retrieval time: {e-s:.3f}")

# print("Top Labels:")
# print(top_labels)
