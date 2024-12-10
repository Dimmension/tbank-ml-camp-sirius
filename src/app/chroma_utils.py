import logging
import json
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from chromadb.config import Settings
from src.config.settings import settings

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class ChromaDBManager:
    def __init__(self, host: str, port: int, embedding_model: str):
        self.host = host
        self.port = port
        self.embedding_model = embedding_model
        self.client = None
        self.embedding_function = None
        self.collection = None
        self.initialize_embedding_function()

    def initialize_client(self):
        """Инициализация клиента ChromaDB."""
        try:
            self.client = chromadb.HttpClient(
                host=self.host,
                port=self.port,
                ssl=False,
                settings=Settings(
                    allow_reset=True,
                    anonymized_telemetry=False
                )
            )
            logging.info(f"ChromaDB client initialized on {self.host}:{self.port}.")
        except Exception as error:
            logging.error(f"Failed to initialize ChromaDB client: {error}")
            raise error

    def initialize_embedding_function(self) -> None:
        """Создание функции эмбеддингов."""
        try:
            self.embedding_function = SentenceTransformerEmbeddingFunction(model_name=self.embedding_model)
            logging.info(f"Embedding function created for model {self.embedding_model}.")
        except Exception as error:
            logging.error(f"Failed to create embedding function: {error}")
            raise error

    def create_chroma_collection(self, collection_name: str) -> None:
        """Создание коллекции ChromaDB."""
        if not self.client:
            raise RuntimeError("ChromaDB client is not initialized. Call `initialize_client` first.")

        if not self.embedding_function:
            raise RuntimeError("Embedding function is not initialized. Call `initialize_embedding_function` first.")

        try:
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                embedding_function=self.embedding_function
            )
            logging.info(f'ChromaDB collection "{collection_name}" initialized.')
        except Exception as error:
            logging.error(f"Failed to create or get collection {collection_name}: {error}")
            raise error

    def add_to_collection(self, path: str):
        with open(path, 'r') as f:
            labels_with_description = json.load(f)
        
        labels = []
        descriptions = []
        metadatas = []

        for label, description in labels_with_description.items():
            labels.append(label)
            descriptions.append(description)
            metadatas.append({"label": label})
        ids = [str(i) for i in range(len(labels))]
        self.collection.add(
            documents=descriptions,
            metadatas=metadatas,
            ids=ids,
        )

if __name__ == '__main__':    
    chroma_manager = ChromaDBManager(
        **settings.CHROMA_SETTINGS
    )
    chroma_manager.initialize_client()
    chroma_manager.create_chroma_collection(settings.COLLECTION_NAME)
    chroma_manager.add_to_collection(settings.DATA_PATH)