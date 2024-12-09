from retrieval import RetrievalSystem
from llm import GeminiAI


# retrieving label candidates 
data_path = r"data\labels_with_description.json"
retrieval_system = RetrievalSystem(data_path, model_name="all-MiniLM-L6-v2")
query = "can you please repeat my list back to me"
label = "reminder"
top_labels = retrieval_system.process_query(query, top_r=10)

print(f"Top labels suggested:\n{top_labels}\n")


# llm calling
model = GeminiAI()
response = model.classify_intent(query=query, dataset_label=label, suggested_intends=top_labels)
print(f"LLM response: {response}\n")
