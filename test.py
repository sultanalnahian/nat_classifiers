import argparse
from inference import Eval
import csv
from tqdm import tqdm
import random   
from infoseeking import InfoSeeking
from relevancy import Relevancy

TASK_CLASSES = {
    'info-seeking' : InfoSeeking,
    'relevancy' : Relevancy
}

FIELD_NAMES = {
    'info-seeking' : ['context', "question", 'original_label', 'predicted_label'],
    'relevancy' : ['context', "question", "history", 'original_label', 'predicted_label']
}

def write_output_to_csv(output_file_name, output, fieldnames):
    with open(output_file_name, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter='\t')
        writer.writeheader()
        for row in output:
            try:
                writer.writerow(row)
            except Exception as e:
                print(e)

def get_infoseeking_output_item(context, target_label, predicted_label):
    item = dict()
    context_pair = context.split("<context>")
    item['context'] = context_pair[1]
    item['question'] = context_pair[0]
    item['original_label'] = target_label
    item['predicted_label'] = predicted_label

    return item

def get_relevancy_output_item(context, target_label, predicted_label):
    item = dict()
    context_pair = context.split("<history>")
    context_pair_2 = context_pair[0].split("<context>")
    item['context'] = context_pair_2[1]
    item['question'] = context_pair_2[0]
    item['history'] = context_pair[1]
    item['original_label'] = target_label
    item['predicted_label'] = predicted_label
    
    return item

def load_data(filePath, task_class):
    nqgOb = task_class(filePath)
    data = nqgOb.data
    inputs = []
    target_labels = []
    random.seed(10)
    random_data = random.sample(data, 100)
    for item in random_data:
        inputs.append(item['context'])
        target_labels.append(item['label'])
        
    return inputs, target_labels

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="relevancy", \
        help='Specify the task. Currently two supported tasks: 1. info-seeking 2. relevancy')
    parser.add_argument("--model", type=str, default="./models/relevancy")
    parser.add_argument("--input_file", type=str, default="data/relevancy_val.json")
    parser.add_argument("--output_file", type=str, default="output/results.tsv")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    task_class = TASK_CLASSES[args.task]
    inputs, target_labels = load_data(args.input_file, task_class)
    print("Generate output for model: ")
    eval = Eval(args.model) 
    output = []
    print("number of data point: ", len(inputs))
    for i, input in tqdm(enumerate(inputs)):
        predicted_label = eval.predict(input)
        if args.task == 'info-seeking':
            item = get_infoseeking_output_item(input, target_labels[i], predicted_label)
        elif args.task == 'relevancy':
            item = get_relevancy_output_item(input, target_labels[i], predicted_label)
        
        output.append(item)

    _field_names = FIELD_NAMES[args.task]
    write_output_to_csv(args.output_file , output, _field_names)
    