from retrieval import RetrievalSystem
from llm import GeminiAI


# retrieving label candidates
data_path = r"data\labels_with_description.json"
retrieval = RetrievalSystem(data_path)

query = "what's a good place to vacation"
label = "travel_suggestion"


# llm calling
model = GeminiAI()

n=3
for i in range(n):
    print(f"Current label: {label}\n")
    top_labels = retrieval.process_query(query, top_r=20)
    top_labels.append(label)
    top_labels_with_descriptions =  {label: retrieval.get_description(label) for label in top_labels}

    print(f"Top labels suggested:\n{top_labels}\n")
    # print(len(top_labels))

    response = model.classify_intent(query=query, dataset_label=label, suggested_intends=top_labels_with_descriptions)
    print(f"LLM response: {response}\n")

    print("-------------------")
    if response != "True":
        label = response

print(f"FINAL LABEL: {label}\n")