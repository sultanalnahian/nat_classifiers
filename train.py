from cmath import inf
import torch
import argparse
# import json
from transformers import T5Config, T5ForConditionalGeneration, T5Tokenizer
from dataset import ModelDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.optim import AdamW
import torch.nn as nn
import pickle
import os
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
import numpy as np

from infoseeking import InfoSeeking
from relevancy import Relevancy

torch.cuda.empty_cache()
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

TASK_CLASSES = {
    'info-seeking' : InfoSeeking,
    'relevancy' : Relevancy
}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataloader_workers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--model", type=str, default="t5-small")
    parser.add_argument("--pad_mask_id", type=int, default=-100)
    parser.add_argument("--pin_memory", dest="pin_memory", action="store_true", default=False)
    parser.add_argument("--save_dir", type=str, default="./models/t5-small")
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--valid_batch_size", type=int, default=16)
    parser.add_argument("--train_data_file", type=str, default="data/train_info.json")
    parser.add_argument("--validation_data_file", type=str, default="data/val_info.json")
    parser.add_argument("--task", type=str, default="info-seeking", \
        help='Specify the task. Currently two supported tasks: 1. info-seeking 2. relevancy')
    
    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()
    tokenizer = T5Tokenizer.from_pretrained(args.model)
    if args.task == 'info-seeking':
        tokenizer.add_special_tokens(
            {'additional_special_tokens': ['<context>', '<question>']}
        )
    elif args.task == 'relevancy':
        tokenizer.add_special_tokens(
            {'additional_special_tokens': ['<context>', '<question>', "<history>", "<q>"]}
        )

    config = T5Config(decoder_start_token_id=tokenizer.pad_token_id)
    model = T5ForConditionalGeneration(config).from_pretrained(args.model)
    model.resize_token_embeddings(len(tokenizer))
    model= nn.DataParallel(model)
    model = model.to(device)

    task_class = TASK_CLASSES[args.task]
    train_data = task_class(args.train_data_file)
    val_data = task_class(args.validation_data_file)
    
    train_set = ModelDataset(train_data.data, args.max_length,  args.pad_mask_id, tokenizer)
    valid_set = ModelDataset(val_data.data, args.max_length, args.pad_mask_id, tokenizer)
    
    train_data_loader = DataLoader(
        train_set, 
        batch_size=args.train_batch_size, 
        num_workers=args.dataloader_workers, 
        pin_memory=args.pin_memory, 
        shuffle=True
        )

    val_data_loader = DataLoader(
        valid_set,
        batch_size=args.valid_batch_size,
        num_workers=args.dataloader_workers,
        pin_memory=args.pin_memory,
        shuffle=False
        )

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    overfit_warn = 0
    min_val_loss = inf
    training_loss_list = []
    val_loss_list = []
    stats_list =[]
    for epoch in range(args.epochs):
        model.train()
        avg_train_loss =0.0
        total_train_loss =0
        with tqdm(total=len(train_data_loader), unit="batches") as tepoch:
            tepoch.set_description(f"epoch {epoch}")
            for data, _ in train_data_loader:
                optimizer.zero_grad()
                data = {key: value.to(device) for key, value in data.items()}
                output = model(**data)
                loss = output.loss
                loss.sum().backward()
                optimizer.step()
                tepoch.set_postfix({"train_loss": loss.sum().item()})
                tepoch.update(1)
                total_train_loss += loss.sum().item()
    
        avg_train_loss = total_train_loss/len(train_data_loader)
        training_loss_list.append(avg_train_loss)
        print("average training_loss: ", avg_train_loss)
        model.eval()
        avg_val_loss = 0
        total_val_loss = 0

        predictions = []
        true_values = []
        with torch.no_grad():
            with tqdm(total=len(val_data_loader), unit="batches") as tepoch:
                tepoch.set_description("validation")
                for data, target_label in val_data_loader:
                    data = {key: value.to(device) for key, value in data.items()}
                    output = model(**data)
                    val_loss = output.loss
                    tepoch.set_postfix({"valid_loss": val_loss.sum().item()})
                    tepoch.update(1)
                    total_val_loss +=val_loss.sum().item()

                    generated_ids = model.module.generate(
                        input_ids=data['input_ids'],
                        attention_mask=data['attention_mask'],
                        max_length=3)

                    preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
                    
                    target = target_label
                    
                    predictions.extend(preds)
                    true_values.extend(target)

            avg_val_loss = total_val_loss/len(val_data_loader)
            val_loss_list.append(avg_val_loss)

            val_precision = precision_score(true_values, predictions, pos_label="yes", average=None)
            val_recall = recall_score(true_values, predictions, pos_label="yes", average=None)
            avg_f1_score = f1_score(true_values, predictions, pos_label="yes", average=None)
            val_acc = accuracy_score(true_values, predictions)
            
            stat_dict = dict()
            stat_dict['precision'] = val_precision
            stat_dict['recall']= val_recall
            stat_dict['f1 score'] = avg_f1_score
            stat_dict['accuracy'] = val_acc
            stat_dict['validation loss'] = avg_val_loss
            stat_dict['training loss'] = avg_train_loss
            stats_list.append(stat_dict)
            
            overfit_warn = overfit_warn + 1 if (min_val_loss < avg_val_loss) else 0
            print("avg validation loss: {} validation acc: {} val f1_score: {} val precision: {} val_recall: {}".format(
                avg_val_loss, val_acc, avg_f1_score, val_precision, val_recall))
            
            if avg_val_loss < min_val_loss:
                print(f"Validation loss decreased from {min_val_loss:.4f} to {avg_val_loss:.4f}. Saving.")
                min_val_loss = avg_val_loss
                model_path = os.path.join(args.save_dir, str(epoch))
                tokenizer.save_pretrained(model_path)
                model.module.save_pretrained(model_path)

        if overfit_warn > 5:
            break

    with open( "training_stats.p", "wb" ) as outputfile:
        pickle.dump(stats_list, outputfile)
        
