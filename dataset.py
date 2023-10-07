import torch
import torch.nn as nn
from torch.utils.data import Dataset, Sampler, Subset
import numpy as np


class BilingualDataset(Dataset):
    def __init__(self, ds, src_tokenizer, tgt_tokenizer, seq_len, src_lang, tgt_lang):
        self.ds = ds
        self.idx = list(range(len(self.ds)))
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer

        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        self.seq_len = seq_len

        self.sos_token = torch.tensor([tgt_tokenizer.token_to_id('[SOS]')], dtype=torch.int64)
        self.eos_token = torch.tensor([tgt_tokenizer.token_to_id('[EOS]')], dtype=torch.int64)
        self.pad_token = torch.tensor([tgt_tokenizer.token_to_id('[PAD]')], dtype=torch.int64)

    def __len__(self):
        return len(self.ds)
    
     
    def clean_data(self):
        for i in range(len(self.ds)):
            src_tgt_pair = self.ds[i]
            src = src_tgt_pair["translation"][self.src_lang]
            tgt = src_tgt_pair["translation"][self.tgt_lang]

            src_tokens = self.src_tokenizer.encode(src).ids
            tgt_tokens = self.tgt_tokenizer.encode(tgt).ids

            if (len(src_tokens) > 150):
                self.idx.remove(i)
            elif (len(tgt_tokens) - len(src_tokens) > 10):
                self.idx.remove(i)
        self.ds = Subset(self.ds, self.idx)
    
    
    def __getitem__(self, idx):
        src_tgt_pair = self.ds[idx]
        src = src_tgt_pair["translation"][self.src_lang]
        tgt = src_tgt_pair["translation"][self.tgt_lang]

        src_tokens = self.src_tokenizer.encode(src).ids
        tgt_tokens = self.tgt_tokenizer.encode(tgt).ids

        src_padding_tokens = self.seq_len - len(src_tokens) - 2
        tgt_padding_tokens = self.seq_len - len(tgt_tokens) - 1

        if (src_padding_tokens < 0) or (tgt_padding_tokens < 0):
            raise ValueError("input seq is too large")
        
        src_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(src_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token]*src_padding_tokens, dtype=torch.int64)
            ],
            dim=0
        )

        tgt_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(tgt_tokens, dtype=torch.int64),
                torch.tensor([self.pad_token]*tgt_padding_tokens, dtype=torch.int64)
            ],
            dim=0
        )

        label = torch.cat(
            [
                torch.tensor(tgt_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token]*tgt_padding_tokens, dtype=torch.int64)
            ],
            dim=0
        )

        assert len(src_input) == self.seq_len
        assert len(tgt_input) == self.seq_len
        assert len(label) == self.seq_len

        return {
            "src_input": src_input,
            "tgt_input": tgt_input,
            "src_mask": (src_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(),
            "tgt_mask": (tgt_input != self.pad_token).unsqueeze(0) & (causal_mask(tgt_input.size(0)).int()),
            "label": label,
            "src_text": src,
            "tgt_text": tgt,
            "src_seq_length": len(src_tokens),
            "tgt_seq_length": len(tgt_tokens)
        }
    
def causal_mask(size):
    mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int64)
    return mask

