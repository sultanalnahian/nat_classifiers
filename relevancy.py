import json
from tqdm import tqdm

class Relevancy:
    def __init__(self, data_file):
        with open(data_file,'rb') as file:
            info_data = json.load(file)
        self.data = self.load_data(info_data)

    def load_data(self, data):
        processed_data = []
        for item in tqdm(data):
            new_item = dict()
            context = item['context']
            question = item['question']
            label_text = item['relevant']
            qa_history = item['qa_history']
            full_context = "<question> {} <context> {} <history>".format(question, context)
            for qa_pair in qa_history:
                full_context += "<q> {}".format(qa_pair['question'])
            
            new_item['context'] = full_context.strip()
            new_item['label'] = label_text
            processed_data.append(new_item)
        
        return processed_data

if __name__ == "__main__":
    info = Relevancy("data/train_500.json")
    print(len(info.data))
