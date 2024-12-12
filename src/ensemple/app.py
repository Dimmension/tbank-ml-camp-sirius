import json
from llm_utils import LLMHandler
from rag_utils import RAGHadler
from retriever_utils import RetrievalSystem

import warnings
warnings.filterwarnings("ignore")

retriever = RetrievalSystem()
llm_model = LLMHandler(
    "bartowski/Qwen2.5-7B-Instruct-GGUF", # 'bartowski/gemma-2-9b-it-GGUF',
    "Qwen/Qwen2.5-7B-Instruct", # 'google/gemma-2-9b-it',
    'Q6_K',
    0.97
)
rag = RAGHadler(retriever, llm_model, 3)

if __name__ == '__main__':
    with open('././data/data_testing_oos.json', 'r') as f:
        data = json.load(f)
    
    result = []
    true = 0
    for label in list(data):
        for query in data[label]:
            top_labels, predicted_label = rag.predict_label(query, label)
            if predicted_label == label:
                true += 1
                
    print(true / len(data))