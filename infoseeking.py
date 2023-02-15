import json
from tqdm import tqdm

class InfoSeeking:
    def __init__(self, data_file):
        with open(data_file,'rb') as file:
            info_data = json.load(file)
        self.data = self.load_data(info_data)

    def load_data(self, data):
        processed_data = []
        for item in tqdm(data):
            new_item = dict()
            context = item['context']
            if "rewritten_question" in item.keys():
                question = item['rewritten_question']
            else:
                question = item['question']
            
            
            label_text = item['answerable']
            full_context = "<question> {} <context> {}".format(question, context)
            
            new_item['context'] = full_context.strip()
            new_item['label'] = label_text
            processed_data.append(new_item)
        
        return processed_data

if __name__ == "__main__":
    info = InfoSeeking("data/train_v2.json")
    print(len(info.data))
