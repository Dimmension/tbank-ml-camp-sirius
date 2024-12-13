import json
import pandas as pd
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
l = 100

if __name__ == '__main__':
    df = pd.read_csv('././data/llm_val_dataset.csv')
    output_path = '././data/val_result.json'
    result = []
    
    for i, row in df.iterrows():
        top_labels, predicted_label = rag.predict_label(row['text'], row['intent'])
        result.append({'query': row['text'], 'initial label': row['true_intent'], 'predicted label': predicted_label})
        
        if i % l == 0:
            with open(output_path, "w") as f:
                json.dump(result, f, indent=4)
    
    with open(output_path, "w") as f:
                json.dump(result, f, indent=4)