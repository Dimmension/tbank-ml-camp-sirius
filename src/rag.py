from retrieval_bge import RetrievalSystem
from gemini import GeminiAI
from jan_model import JanModel
import random
import time


# retrieving label candidates
retrieval = RetrievalSystem()
# model = GeminiAI()
model = JanModel()


def predict_label(query, label, n: int=3):
    for _ in range(n):
        print(f"Current label: {label}\n")
        top_labels = retrieval.process_query(query, top_r=30, top_m=15)

        # check if query is out of domain example
        if len(top_labels) != 0:
            if label == "oos":
                label = random.choice(retrieval.get_labels())
            if label not in top_labels:
                top_labels.append(label)
                
            top_labels_with_descriptions =  {label: retrieval.get_description(label) for label in top_labels}
            response = model.classify_intent(query=query, suggested_intends=top_labels_with_descriptions)
            label = response

            print(f"Top labels suggested:\n{top_labels}\n")
            print(f"LLM response: {response}\n")
            print("-------------------")
        else:
            label = "oos"
            print("Out of Domain example")

    print(f"FINAL LABEL: {label}\n")
    return label


# exmaples of usage

# query = "what's a good place to vacation"
# label = "travel_suggestion"

query = "in what month does my credit card expire"
label = "travel_suggestion"

# query = "what years has korea been at war"
# label = "expiration_date"

# query = "who formulated the theory of relativity"
# label = "travel_suggestion"


s = time.time()
answer = predict_label(query, label)
print(f"Time: {time.time() - s}")
