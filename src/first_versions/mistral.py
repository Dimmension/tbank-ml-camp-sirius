from dotenv import load_dotenv
import os
from mistralai import Mistral


class MistralAI:
    def __init__(self):
        load_dotenv()
        self.model = "open-mistral-7b"
        self.client = Mistral(api_key=os.getenv('MISTRAL_API_KEY'))

    def generate_response(self, request, relevant_info=""):
        prompt = f"User request: {request}\n\nUser provided link from where relevant information was extracted.\
                Answer user request only based on the most relevant link content from this: \n{relevant_info}\
                If request is not related to provided content you should deny to provide answer to that request.\
                Be concise, provide only the answer on the question."

        chat_response = self.client.chat.complete(
            model=self.model,
            messages = [
                {
                    "role": "user",
                    "content": prompt,
                },
            ]
        )
        return chat_response.choices[0].message.content