import json
import pandas as pd

with open("../data/data_full_spoiled.json", "r") as f:
    data = json.load(f)

d = {"text": [], "intent": []}

for text, intent in data["train"]:
    d["text"].append(text)
    d["intent"].append(intent)

for text, intent in data["oos_train"]:
    d["text"].append(text)
    d["intent"].append(intent)

pd.DataFrame(d).to_csv("../data/llm_train_dataset.csv", index=False)
