import json
import pandas as pd
from llm_utils import LLMHandler
from rag_utils import RAGHadler
from retriever_utils import RetrievalSystem
import time
import warnings
warnings.filterwarnings("ignore")

retriever = RetrievalSystem()
llm_model = LLMHandler(
    llm_model_name="bartowski/Qwen2.5-7B-Instruct-GGUF",
    llm_tokenizer_name="Qwen/Qwen2.5-7B-Instruct",
    gguf_q_type='Q6_K',
    threshhold=0.97
)
rag = RAGHadler(retriever=retriever, llm=llm_model, count_is_again=3)
save_step = 100

if __name__ == '__main__':

    df = pd.read_csv('././data/llm_train_dataset.csv')
    output_path = '././data/train_result_llama.json'
    result = []
    
    for i, row in df.iterrows():
        top_labels, predicted_label = rag.predict_label(row['text'], row['intent'])
        result.append({'query': row['text'], 'initial label': row['intent'], 'predicted label': predicted_label})
        
        if i % save_step == 0:
            with open(output_path, "w") as f:
                json.dump(result, f, indent=4)
    
    with open(output_path, "w") as f:
        json.dump(result, f, indent=4)
        
