import requests
import json
import pandas as pd


class JanModel:
    def __init__(self):
        self.url = "http://127.0.0.1:1337/v1/chat/completions"
        self.headers = {
            "Content-Type": "application/json"
        }
    
    def classify_intent(self, query, suggested_intends):
        system_prompt = f"""
        You are an advanced AI designed to annotate user intends (label) for queries
        Choose the most appropriate label from the defined set of intents and respond with that label.
        Every intent is provided with its description and the example of context in which this label may be used.

        If none of the labels match the query with a sufficient confidence considering the labels with their
        descriptions, return "oos" label meaning out of domain text.
        Return only name of the correct label, nothing else!!

        Defined set of the intends with their descriptions and examples: {suggested_intends}
        """
        user_prompt = f"Define user intend for the following query.\nQuery: {query}\nIntent:"
        # print(user_prompt)


        payload = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "model": "llama3.1-8b-instruct",
            # "model": "openchat-3.5-7b",
            "stream": False,
            # "context_length": 60000,
            # "max_tokens": 8000,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "temperature": 0.3,
            "top_p": 0.9,
        }

        try:
            response = requests.post(self.url, headers=self.headers, data=json.dumps(payload))
            if response.status_code == 200:
                result = response.json()
                content = result['choices'] [0] ['message'] ['content']
                possible_label = list(suggested_intends.keys())
                print(f"--content: {content}")
                if content != "oos" and not (content in possible_label):
                    return self.classify_intent(query, suggested_intends)
                
                return content
            else:
                print(f"Error: {response.status_code}, {response.text}")
        except requests.exceptions.RequestException as e:
            print("Request failed:", e)

