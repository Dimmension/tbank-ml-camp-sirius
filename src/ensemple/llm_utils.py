import os
import logging
from llama_cpp import Llama
from transformers import AutoTokenizer
from dotenv import load_dotenv


# logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

class LLMHandler:
    load_dotenv()
    CACHE_PATH = '/home/student/aboba/workspace/hf_cache'
    TOKEN_HF = os.getenv('HF_TOKEN')
    GENERATION_CONFIG = {
        'max_tokens': 100,
        'echo': False,
        'temperature': 0.2,
        'top_k': 40,
        'top_p': 0.95,
        'min_p': 0.3,
        'logprobs': 1,
    }

    def __init__(
        self,
        llm_model_name,
        llm_tokenizer_name,
        gguf_q_type,
    ) -> None:        
        self.llm_model = Llama.from_pretrained(
            repo_id=llm_model_name,
            filename=f"*{gguf_q_type}.gguf",
            verbose=True,
            cache_dir=self.CACHE_PATH,
            n_ctx=8192,
            n_gpu_layers=-1,
            logits_all=True,
        )

        self.llm_tokenizer = AutoTokenizer.from_pretrained(
            llm_tokenizer_name,
            cache_dir=self.CACHE_PATH,
            token=self.TOKEN_HF,
        )

    def generate(
        self,
        query: str,
        suggested_intents: str
    ) -> tuple:
        system_prompt = f"""
            You are an advanced AI designed to annotate user intends (label) for queries
            Choose the most appropriate label from the defined set of intents and respond with that label.
            Every intent is provided with its description and the example of context in which this label may be used.
            Return only name of the correct label. If nothing suits, return "oos" label meaning out of domain text!
            
            Defined set of the intends with their descriptions and examples: {suggested_intents}
        """
        user_prompt = f"{system_prompt}\nQuery: {query}\nIntent:"

        history = [
            {'role': 'user', 'content': user_prompt}
        ]

        inputs = self.llm_tokenizer.apply_chat_template(
            conversation=history,
            add_generation_prompt=True,
            tokenize=False,
        )

        outputs = self.llm_model(
            inputs,
            **self.GENERATION_CONFIG,
        )
        generated_text = outputs['choices'][0]['text']

        token_probs = outputs['choices'][0]['logprobs']['token_logprobs']
        tokens = ' '.join(outputs['choices'][0]['logprobs']['tokens'])

        confidences = [round(10 ** prob, 4) for prob in token_probs]
        logging.warning(f"TOKEN: {tokens[0]}\t CONFIDENCE: {confidences}")
        return generated_text