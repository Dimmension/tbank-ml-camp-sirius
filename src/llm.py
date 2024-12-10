import google.generativeai as genai
from dotenv import load_dotenv
import os
import time


with open(r"data\labels_with_description.json") as file:
    descriptions = pd.read_csv('data\label_desc_context.csv')
    LABELS_DESCRIPTIONS = []
    for i, row in descriptions.iterrows():
        LABELS_DESCRIPTIONS.append({'intent': row['label'], 'description': row['intent'], 'context': row['text']})


class GeminiAI:
    def __init__(self):
        load_dotenv()
        genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
        self.model = genai.GenerativeModel(model_name="gemini-1.5-flash")
        
    def classify_intent(self, query, dataset_label, suggested_intends=LABELS):
        # time.sleep(5)
        system_prompt = f"""
        You are an advanced AI designed to annotate user intends (label) for queries\n
        Choose the most appropriate label 
        from the defined set of intents and respond with that label.
        Every intent is provided with its description and the example of context in which this label may be used
        
        Defined set of the intends with their descriptions and examples: {suggested_intends}
        """
        
        user_prompt = f"{system_prompt}\n\nQuery: {query}\nIntent:"
        
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

