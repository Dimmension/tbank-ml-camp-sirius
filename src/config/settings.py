from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    CHROMA_SETTINGS: dict = {
        'host': 'localhost',
        'port': 4810,
        'embedding_model': 'cointegrated/rubert-tiny2',
    }
    COLLECTION_NAME: str = 'tbank_rag'
    DATA_PATH: str = './data/labels_with_description.json'
    class Config:
        env_file = 'config/.env'


settings = Settings()