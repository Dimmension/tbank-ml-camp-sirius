import random
import logging

logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

class RAGHadler:
    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm

    def predict_label(self, query, label, n: int=2):
        logging.warning(f"QUERY:\t{query}")
        logging.warning(f"EXPECTED LABEL:\t{label}")
        for _ in range(n):
            top_labels = self.retriever.process_query(query, top_r=30, top_m=30)

            if len(top_labels) != 0:
                if label == "oos":
                    label = random.choice(self.retriever.get_labels())
                if label not in top_labels:
                    top_labels.append(label.strip('\n').strip())
                    
                top_labels_with_descriptions = {label: self.retriever.get_description(label) for label in top_labels}
                response = self.llm.generate(query, top_labels_with_descriptions)
                label = response
            else:
                label = "oos"
                top_labels_with_descriptions = {}
        logging.warning(f"FINAL LABEL:\t{label}")
        return top_labels_with_descriptions, label
