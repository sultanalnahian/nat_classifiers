import json
import random
from random import choice
from nltk import tokenize
from tqdm import tqdm
import argparse

def get_answer_position(context, answer):
    start = 0
    end = 0
    splitted_context = context.split(answer)
    
    left_c_sentences = tokenize.sent_tokenize(splitted_context[0])
    right_c_sentences = tokenize.sent_tokenize(splitted_context[1])
    answer_sentences = tokenize.sent_tokenize(answer)
    c_sentences = tokenize.sent_tokenize(context)
    left_position = len(left_c_sentences)-1
    start = left_position+1
    if answer_sentences[0] in c_sentences[left_position]:
        start = left_position
    end = start+len(answer_sentences)
        
    return start, end

def get_nonrelevant_question(skip_item, data):
    data_len = len(data)
    skip_items = [i for i in range(skip_item-2, skip_item+2)]
    random_item_no = choice([i for i in range(0,data_len) if i not in skip_items])
    random_paragraph = data[random_item_no]['paragraphs'][0]
    qas = random_paragraph['qas']
    qas_no = random.randint(0,len(qas)-1)
    question = qas[qas_no]['question']
    return question

def insert_data(contexts, question, answer, qa_history, relevant):
    data_points = []
    for context in contexts:
        if context.strip() != "" and context != 'CANNOTANSWER':
            data_item = dict()
            data_item['context'] = context
            data_item['question'] = question
            data_item['answer'] = answer
            data_item['qa_history'] = qa_history
            data_item['relevant'] = relevant
            data_points.append(data_item)

    return data_points

def generate_data_from_paragraph(paragraph, item_no, original_data):
    data_points = []
    context = paragraph['context']
    c_sentences = tokenize.sent_tokenize(context)
    qas = paragraph['qas']
    qa_history = []
    for qa in qas:
        answer_text = qa['orig_answer']['text']
        question = qa['question']

        if answer_text != 'CANNOTANSWER':
            start_p, end_p = get_answer_position(context, answer_text)
            trimmed_context = ' '.join(c_sentences[:start_p])
            if trimmed_context.strip() != "":
                data_points += insert_data([trimmed_context], question, answer_text, qa_history.copy(), 'yes')

                non_relevant_question = get_nonrelevant_question(item_no, original_data)
                
                data_points += insert_data([trimmed_context], non_relevant_question, "", qa_history.copy(), 'no')

        qa_history_item = dict()
        qa_history_item['question'] = question
        qa_history_item['answer'] = answer_text
        qa_history.append(qa_history_item)
    
    return data_points

def generate_data(source_file_path, output_file_path):
    with open(source_file_path, encoding="utf8") as source_file:
        _data = json.load(source_file)
    data = _data['data']
    total_data_set = []
    for item_no, data_item in tqdm(enumerate(data)):
        paragraphs = data_item['paragraphs']
        for item in paragraphs:
            generated_data = generate_data_from_paragraph(item, item_no, data)
            total_data_set += generated_data

    with open(output_file_path, "w") as output_file:
        json.dump(total_data_set, output_file)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_file", type=str, default="source_data/QuAC/train_v0.2.json")
    parser.add_argument("--output_file", type=str, default="data/train_info.json")
    return parser.parse_args()    

if __name__ == "__main__":
    args = parse_args()
    generate_data(args.source_file, args.output_file)


