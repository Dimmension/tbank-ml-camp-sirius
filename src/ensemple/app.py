import json
import logging
import warnings
from llm_utils import LLMHandler
from rag_utils import RAGHadler
from retriever_utils import RetrievalSystem

warnings.filterwarnings("ignore", category=RuntimeWarning)

retriever = RetrievalSystem()
llm_model = LLMHandler(
    'bartowski/gemma-2-9b-it-GGUF',
    'google/gemma-2-9b-it',
    'Q8_0',
)
rag = RAGHadler(retriever, llm_model)

if __name__ == '__main__':
    with open('./data/data_full_spoiled.json', 'r') as f:
        data = json.load(f)
    
    result_val = []
    for candidate in data['val']:
        text = candidate[0]
        label = candidate[1]
        top_labels, predicted_label = rag.predict_label(text, label)
        result_val.append({'top_labels': [k for k in top_labels.keys()], 'predicted_label': label})

    result_oos_val = []
    for candidate in data['oos_val']:
        text = candidate[0]
        label = candidate[1]
        top_labels, predicted_label = rag.predict_label(text, label)
        result_oos_val.append({'top_labels': [k for k in top_labels.keys()], 'predicted_label': label})

    result_data = {'val': result_val, 'oos_val': result_oos_val}
    with open('./data/predicted_val_labels.json', "w") as file:
        json.dump(result_data, file, indent=4)
