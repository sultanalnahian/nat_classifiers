import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)

class Eval:

    def __init__(self, pretrained_model='t5-base-eval') -> None:

        self.SEQ_LENGTH = 512

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model, use_fast=False)
        self.nqg_model = AutoModelForSeq2SeqLM.from_pretrained(pretrained_model)
        self.nqg_model.to(self.device)
        self.nqg_model.eval()

    def predict(
        self,
        context: str):

        encoded_input = self.tokenizer(
            context,
            padding='max_length',
            max_length=self.SEQ_LENGTH,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.nqg_model.generate(input_ids=encoded_input["input_ids"], max_length=3)

        label = self.tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
            
        return label

# if __name__ == "__main__":
    # eval = Eval("./models/infoseeking")
    # prediction = eval.predict("<question> what is the largest river basin in the world? <context> Amazonia is the largest river basin in the world, and its forest stretches from the Atlantic Ocean in the east to the tree line of the Andes in the west.")
    # print(prediction)