import google.generativeai as genai
from dotenv import load_dotenv
import os
import time


with open(r"data\classes.txt", "r") as file:
    LABELS =  [line.strip() for line in file.readlines() if line.strip()]


class GeminiAI:
    def __init__(self):
        load_dotenv()
        genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
        self.model = genai.GenerativeModel(model_name="gemini-1.5-flash")
        
    def classify_intent(self, query, dataset_label, suggested_intends=LABELS):
        # time.sleep(5)
        system_prompt = f"""
        You are an advanced AI designed to annotate user intends (label) for queries.
        Follow these steps to process the query:\n
        1. Verify if the provided label matches the query meaningfully and is part of 
        the defined set of intents. If the label is correct, respond with 'True'.
        2. If the provided label is incorrect, choose the most appropriate label 
        from the defined set of intents and respond with that label.
        3. If the query does not fit any label in the defined set of intents, 
        respond with 'oos', meaning 'out of scope'.

        Defined set of the intends: {suggested_intends}
        """
        
        user_prompt = f"{system_prompt}\n\nQuery: {query}\nLabel: {dataset_label}\nIntent:"
        
        try:
            response = self.model.generate_content(user_prompt)
            return response.text.strip()
        except Exception as e:
            print(f"Error generating intent: {e}")
            return None


# exmaple

# model = GeminiAI()
# response = model.classify_intent("in french, how do i say, see you later", "translate")
# print(response)

