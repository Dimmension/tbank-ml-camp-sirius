import os
import logging
from llama_cpp import Llama
from transformers import AutoTokenizer

os.environ['CUDA_VISIBLE_DEVICES'] = '2'
MODEL_CACHE='/home/root/hf_cache'
TOKEN_HF='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'

LLM_GGUF_MODEL_NAME='bartowski/gemma-2-9b-it-GGUF'
LLM_MODEL_TOKENIZER_NAME='google/gemma-2-9b-it'

GGUF_Q_TYPE='Q8_0'
GENERATION_CONFIG = {
    'max_tokens': 2048,
    'echo': False,
    'temperature': 0.2,
    'top_k': 40,
    'top_p': 0.95,
    'min_p': 0.3
}

logging.info(MODEL_CACHE)
logging.info(f"CUDA VISIBLE DEVICES: {os.getenv('CUDA_VISIBLE_DEVICES')}")

llm_model = Llama.from_pretrained(
    repo_id=LLM_GGUF_MODEL_NAME,
    filename=f"*{GGUF_Q_TYPE}.gguf",
    verbose=True,
    cache_dir=MODEL_CACHE,
    n_ctx=8192,
    n_gpu_layers=-1
)

llm_tokenizer = AutoTokenizer.from_pretrained(
    LLM_MODEL_TOKENIZER_NAME,
    cache_dir=MODEL_CACHE,
    token=TOKEN_HF,
)


def generate( 
    query: str,
    suggested_intents: str
) -> tuple:
    system_prompt = f"""
        You are an advanced AI designed to annotate user intents (label) for queries\n
        Choose the most appropriate label from the defined set of intents and respond with that label.
        Every intent is provided with its description and the example of context in which this label may be used.
        Return only name of the correct label, nothing else!
        
        Defined set of the intents with their descriptions and examples: {suggested_intents}
        """
    user_prompt = f"{system_prompt}\nQuery: {query}\nIntent:"

    history = [
        {'role': 'user', 'content': user_prompt}
    ]

    inputs = llm_tokenizer.apply_chat_template(
        conversation=history,
        add_generation_prompt=True,
        tokenize=False,
    )

    outputs = llm_model(
        inputs,
        **GENERATION_CONFIG,
    )['choices'][0]['text']

    print(f'GENERATED LABEL: {outputs}')
    return outputs
