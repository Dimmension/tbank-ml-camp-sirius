import random
import numpy as np


class RAGHadler:
    def __init__(self, retriever, llm, k):
        self.retriever = retriever
        self.llm = llm
        self.k = k

    def predict_label(self, query, target):
        is_again = True
        print(f"EXPECTED: {target}")
        top_labels = self.retriever.process_query(query, top_r=20, top_k=10)

        if len(top_labels) != 0:
            if target == "oos": target = random.choice(top_labels)
            if target not in top_labels: top_labels.append(target)
                
            nearest_labels = self.retriever.get_nearest_labels(target)
            for near_label in nearest_labels:
                if near_label not in top_labels:
                    top_labels.append(near_label)
                    
            top_labels_with_descriptions = {label: self.retriever.get_description(label) for label in top_labels}
            
            predicted = {}
            count, mean_min_score = 0, 0
            while is_again:
                target, is_again, min_score = self.llm.generate(query, top_labels_with_descriptions, target)
                
                if is_again:
                    if target == "oos": target = random.choice(top_labels)

                    count, mean_min_score = count + 1, mean_min_score + min_score
                    if target in predicted: predicted[target] += 1
                    else: predicted[target] = 1
                    
                    if count == self.k:
                        mean_min_score /= self.k
                        target = self.find_best(predicted, mean_min_score)
                        break

        else:
            target = "oos"
            top_labels_with_descriptions = {}

        print(f"PREDICTED: {target}")
        return top_labels_with_descriptions, target
    
    def find_best(self, predicted, mean_min_score):
        n = len(predicted)
        if n == self.k or mean_min_score <= 0.5: return "oos"
        
        labels, values = list(predicted), np.array(list(predicted.values()))
        return labels[np.argmax(values)]
    