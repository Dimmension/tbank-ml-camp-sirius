from generation_llm import GeminiAI
import pandas as pd


model = GeminiAI()

def calculate_accuracy(data, ai_model, intents):
    """
    Calculate the accuracy of label classification for a dataset.
    
    :param data: DataFrame containing 'label' and 'query' columns
    :param ai_model: Instance of GeminiAI
    :param intents: List of intents to be passed to the model
    :return: Accuracy as a percentage
    """
    correct_count = 0
    total = len(data)
    
    for _, row in data.iterrows():
        label = row['label']
        query = row['val']
        response = ai_model.classify_intent(query, label)
        print(f"------------\n{query}\n{label}\n{response}\n")
        
        if response == 'True':
            correct_count += 1
    
    accuracy = (correct_count / total) * 100
    return accuracy


df = pd.read_csv("df_check.csv")

intents = df['label']
intents_list = "\n".join(intents)


val_accuracy = calculate_accuracy(df[['label', 'val']], model, intents_list)
print(f"Validation Accuracy: {val_accuracy:.2f}%\n\nXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")


train_data = df[['label', 'train']].rename(columns={'train': 'val'})
train_accuracy = calculate_accuracy(train_data, model, intents_list)
print(f"Training Accuracy: {train_accuracy:.2f}%")

