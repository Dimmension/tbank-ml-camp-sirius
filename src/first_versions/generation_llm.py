import google.generativeai as genai
from dotenv import load_dotenv
import os
import time
import pandas as pd


class GeminiAI:
    def __init__(self):
        load_dotenv()
        genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
        self.model = genai.GenerativeModel(model_name="gemini-1.5-flash")
        
    def classify_intent(self, query, suggested_intends):
        time.sleep(5)
        system_prompt = f"""
        You are an advanced AI designed to annotate user intends (label) for queries
        Choose the most appropriate label from the defined set of intents and respond with that label.
        Every intent is provided with its description and the example of context in which this label may be used.
        Return only name of the correct label. If nothing suits, return "oos" label meaning out of domain text!
        
        Defined set of the intends with their descriptions and examples: {suggested_intends}
        """
        user_prompt = f"{system_prompt}\n\nQuery: {query}\nIntent:"
        
        try:
            response = self.model.generate_content(user_prompt)
            response = response.text.strip()
            possible_label = list(suggested_intends.keys())

            if not response in possible_label:
                return self.classify_intent(query, suggested_intends)
            return response
        
        except Exception as e:
            print(f"Error generating intent: {e}")
            return None


# example

# model = GeminiAI()
# response = model.classify_intent("in french, how do i say, see you later", "translate")
# print(response)

