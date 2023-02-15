import torch
from transformers import AutoTokenizer
from typing import Mapping, Tuple

class ModelDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data,
        max_length: int,
        pad_mask_id: int,
        tokenizer: AutoTokenizer
    ) -> None:
        self.data = data
        self.max_length = max_length
        self.pad_mask_id = pad_mask_id
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Mapping[str, torch.Tensor]:
        item = self.data[index]
        input_ids, attention_mask = self._encode_text(item['context'],self.max_length)
        labels, label_attention_mask = self._encode_text(item['label'],3)
        masked_labels = self._mask_label_padding(labels)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": masked_labels
            # "decoder_attention_mask":label_attention_mask
        }, item['label']

    def _encode_text(self, text: str, max_length) -> Tuple[torch.Tensor, torch.Tensor]:
        encoded_text = self.tokenizer(
            text,
            add_special_tokens=True,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        return (
            encoded_text["input_ids"].squeeze(),
            encoded_text["attention_mask"].squeeze()
        )

    def _mask_label_padding(self, labels: torch.Tensor) -> torch.Tensor:
        labels[labels == self.tokenizer.pad_token_id] = self.pad_mask_id
        return labels
