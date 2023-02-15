import argparse
import json
import random
from nltk import tokenize
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

qrw_tokenizer = AutoTokenizer.from_pretrained("castorini/t5-base-canard")
qrw_model = AutoModelForSeq2SeqLM.from_pretrained("castorini/t5-base-canard")
qrw_model.to(device)

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

def generate_answerable_contexts(context_sents, ans_pos_start, ans_pos_end):
    left_context = ' '.join(context_sents[:ans_pos_end])
    # right_context = ' '.join(context_sents[ans_pos_start:])
    right_position = min(len(context_sents), ans_pos_end + random.randint(1,4))
    left_position = max(0, ans_pos_start - random.randint(1,7))
    
    
    center_context = ' '.join(context_sents[left_position:right_position])
    contexts =[left_context]
    
    if left_context != center_context:
        contexts.append(center_context)
    return contexts

def generate_nonanswerable_contexts(context_sents, ans_pos_start, ans_pos_end):
    left_context = ' '.join(context_sents[:ans_pos_start])
    right_context = ' '.join(context_sents[ans_pos_end:])
    contexts = [left_context, right_context]
    
    return contexts

def get_self_contained_question(context, answer, question):
    c_sentences = tokenize.sent_tokenize(context)
    if answer == 'CANNOTANSWER':
        answer = ""
        end_p = len(c_sentences) - 1
    else:
        start_p, end_p = get_answer_position(context, answer)

    trimmed_context = ' '.join(c_sentences[:end_p])
    source_text = "{} ||| {} ||| {}".format(trimmed_context, answer, question)
    
    inputs = qrw_tokenizer(source_text, return_tensors="pt").to(device)
    output = qrw_model.generate(**inputs)
    rewritten_question = qrw_tokenizer.decode(output[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return rewritten_question

def insert_data(contexts, question, rewritten_ques, answer, qa_history, answerable):
    data_points = []
    for context in contexts:
        if context.strip() != "" and context != 'CANNOTANSWER':
            data_item = dict()
            data_item['context'] = context
            data_item['question'] = question
            data_item['rewritten_question'] = rewritten_ques
            data_item['answer'] = answer
            data_item['qa_history'] = qa_history
            data_item['answerable'] = answerable
            data_points.append(data_item)

    return data_points

def get_random_context(context):
    max_sent = 10
    start_p = random.randint(0, max(0, len(context)-max_sent))
    new_context = ' '.join(context[start_p:min(start_p+max_sent, len(context))])
    return new_context

def generate_data_from_paragraph(paragraph):
    data_points = []
    context = paragraph['context']
    c_sentences = tokenize.sent_tokenize(context)
    qas = paragraph['qas']
    qa_history = []
    for qa in qas:
        answer_text = qa['orig_answer']['text']
        question = qa['question']
        rewritten_question = get_self_contained_question(context, answer_text, question)
        if answer_text == 'CANNOTANSWER':
            data_points += insert_data([get_random_context(c_sentences)], question, rewritten_question, answer_text, qa_history.copy(), 'no')
        else:
            start_p, end_p = get_answer_position(context, answer_text)
            answerable_contexts = generate_answerable_contexts(c_sentences, start_p, end_p)
            non_answerable_contexts = generate_nonanswerable_contexts(c_sentences, start_p, end_p)
            data_points += insert_data(answerable_contexts, question, rewritten_question, answer_text, qa_history.copy(), 'yes')
            data_points += insert_data(non_answerable_contexts, question, rewritten_question, answer_text, qa_history.copy(), 'no')

        qa_history_item = dict()
        qa_history_item['question'] = question
        qa_history_item['answer'] = answer_text
        qa_history.append(qa_history_item)
    
    return data_points

def generate_infoseeking_data(source_file_path, output_file_path):
    with open(source_file_path, encoding="utf8") as source_file:
        _data = json.load(source_file)
    
    data = _data['data']
    total_data_set = []
    
    for data_item in tqdm(data):
        paragraphs = data_item['paragraphs']
        for item in paragraphs:
            generated_data = generate_data_from_paragraph(item)
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
    generate_infoseeking_data(args.source_file, args.output_file)


