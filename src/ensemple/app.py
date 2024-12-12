import json
import logging
import warnings
from llm_utils import LLMHandler
from rag_utils import RAGHadler
from retriever_utils import RetrievalSystem

warnings.filterwarnings("ignore", category=RuntimeWarning)

retriever = RetrievalSystem()
llm_model = LLMHandler(
    "bartowski/Qwen2.5-7B-Instruct-GGUF", # 'bartowski/gemma-2-9b-it-GGUF',
    "Qwen/Qwen2.5-7B-Instruct", # 'google/gemma-2-9b-it',
    'Q6_K',
    0.9
)
rag = RAGHadler(retriever, llm_model, 3)

if __name__ == '__main__':
    with open('././data/data_testing.json', 'r') as f:
        data = json.load(f)
    
    result = []
    k = 3
    for label in list(data):
        for query in data[label][:k]:
            top_labels, predicted_label = rag.predict_label(query, label)
        # result.append({'top_labels': [k for k in top_labels.keys()], 'predicted_label': label})

    # result_data = {'val': result}
    # with open('././data/predicted_val_labels.json', "w") as file:
    #     json.dump(result_data, file, indent=4)
