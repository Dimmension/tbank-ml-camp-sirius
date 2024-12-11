import random
import json
from retrieval import RetrievalSystem
from llm import generate
data_path = "./data/labels_with_description.json"
retrieval = RetrievalSystem(data_path)


def predict_label(query, label, n: int=2):
    for _ in range(n):
        print(f"QUERY:\t{query}\n")
        print(f"EXPECTED LABEL:\t{label}\n")
        top_labels = retrieval.process_query(query, top_r=30, top_m=30)

        # check if query is out of domain example
        if len(top_labels) != 0:
            if label == "oos":
                label = random.choice(retrieval.get_labels())
            if label not in top_labels:
                top_labels.append(label.strip('\n').strip())
                
            top_labels_with_descriptions = {label: retrieval.get_description(label) for label in top_labels}
            response = generate(query=query, suggested_intends=top_labels_with_descriptions)
            label = response

            # print(f"Top labels suggested:\n{top_labels}\n")
            # print(f"LLM response: {response}\n")
            # print("-------------------")
        else:
            label = "oos"
            top_labels_with_descriptions = {}
            # print("Out of Domain example")

    print(f"FINAL LABEL:\t{label}\n")
    return top_labels_with_descriptions, label

if __name__ == '__main__':
    with open('./data/data_full_spoiled.json', 'r') as f:
        data = json.load(f)
    
    result_val = []
    for candidate in data['val']:
        text = candidate[0]
        label = candidate[1]
        top_labels, label = predict_label(text, label)
        result_val.append({'top_labels': [k for k in top_labels.keys()], 'predicted_label': label})

    result_oos_val = []
    for candidate in data['oos_val']:
        text = candidate[0]
        label = candidate[1]
        top_labels, label = predict_label(text, label)
        result_oos_val.append({'top_labels': top_labels, 'predicted_label': label})

    result_data = {'val': result_val, 'oos_val': result_oos_val}
    with open('./data/predicted_val_labels.json', "w") as file:
        json.dump(result_data, file, indent=4)

# exmaples of usage

# query = "what's a good place to vacation"
# label = "travel_suggestion"

# query = "in what month does my credit card expire"
# label = "travel_suggestion"

# query = "what years has korea been at war"
# label = "expiration_date"

# query = "who formulated the theory of relativity"
# label = "travel_suggestion"

# answer = predict_label(query, label)