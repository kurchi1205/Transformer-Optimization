from model import build_transformer
from dataset import BilingualDataset, causal_mask
from config import get_config, get_weight_file_path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim import lr_scheduler
# from torchtext.datasets import datasets

import warnings
from tqdm import tqdm
import os 
from pathlib import Path
import torchmetrics
import pytorch_lightning as pl

warnings.filterwarnings("ignore")

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace


def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    encoder_output = model.encode(source, source_mask)

    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
    while(True):
        if (decoder_input.size(1) == max_len):
            break

        decoder_output = model.decode(encoder_output, source_mask, decoder_input, causal_mask(decoder_input.size(1)).type_as(source_mask).to(device))
        prob = model.project(decoder_output[:, -1])
        _, next_token = torch.max(prob, dim=-1)
        decoder_input = torch.cat([decoder_input, torch.empty(1, 1).type_as(source).fill_(next_token.item())], dim=1)

        if (next_token.item() == eos_idx):
            break

    return decoder_input.squeeze(0)


def get_all_sentences(ds, lang):
    for item in ds:
        yield item["translation"][lang]


def get_or_build_tokenizer(config, ds, lang):
    tokenizer_file = Path(config["tokanizer_file"].format(lang))
    if (tokenizer_file.exists()):
        tokenizer = Tokenizer.from_file(str(tokenizer_file))
    else:
        tokenizer = Tokenizer(WordLevel(unk_token='[UNK]'))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=['[UNK]', '[PAD]', '[SOS]', '[EOS]'], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_file))
    return tokenizer


def get_ds(config):
    ds_raw = load_dataset("opus_books", f"{config['lang_src']}-{config['lang_tgt']}", split="train")
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config["lang_src"])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config["lang_tgt"])

    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = torch.utils.data.random_split(ds_raw, [train_ds_size, val_ds_size])

    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config["seq_len"], config["lang_src"], config["lang_tgt"])
    val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config["seq_len"], config["lang_src"], config["lang_tgt"])
    
    max_len_src = 0
    max_len_tgt = 0

    for item in ds_raw:
        src_ids = tokenizer_src.encode(item["translation"][config["lang_src"]]).ids
        tgt_ids = tokenizer_tgt.encode(item["translation"][config["lang_tgt"]]).ids

        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f"max_len_src: {max_len_src}")
    print(f"max_len_tgt: {max_len_tgt}")

    train_dataloader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True, num_workers=4)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt


def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_transformer(vocab_src_len, vocab_tgt_len, config["seq_len"], config["seq_len"], config["d_model"], config["N"], config["head"], config["dropout"], config["d_ff"])
    return model


class CustomLightningModule(pl.LightningModule):
    def __init__(self, config, model, tokenizer_src, tokenizer_tgt):
        super().__init__()
        self.config = config
        self.model = model
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Path(config['model_folder']).mkdir(parents=True, exist_ok=True)
        # self.train_loader, self.val_loader, self.tokenizer_src, self.tokenizer_tgt = get_ds(config)
        # self.model = get_model(config, self.tokenizer_src.get_vocab_size(), self.tokenizer_tgt.get_vocab_size())

        # if config["preload"]:
        #     model_filename = get_weight_file_path(config, config["preload"])
        #     print("Preloading model ", model_filename)
        #     state_dict = torch.load(model_filename)
        #     self.model.load_state_dict(state_dict)
        #     self.initial_epoch = state_dict['epoch'] + 1
        #     # self.optimizer.load_state_dict(state_dict) 
        #     self.global_step = state_dict['global_step'] + 1
        #     print("Model preloaded")
    
        # model = model
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=self.tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1)


    def forward(self, src_input, src_mask, tgt_input, tgt_mask):
        encoder_output = self.model.encode(src_input, src_mask)
        decoder_output = self.model.decode(tgt_input, encoder_output, src_mask, tgt_mask)
        proj_output = self.model.project(decoder_output)
        return proj_output
    

    def training_step(self, batch):
        src_input = batch["src_input"]
        tgt_input = batch["tgt_input"]
        src_mask = batch["src_mask"]
        tgt_mask = batch["tgt_mask"]

        proj_output = self.forward(src_input, src_mask, tgt_input, tgt_mask)

        label = batch["label"].to(self.device)
        loss = self.loss_fn(proj_output.view(-1, self.tokenizer_tgt.get_vocab_size()), label.view(-1, ))
        self.log('train_loss', loss)
        return loss
    

    def validation_step(self, batch, batch_idx):
        encoder_input = batch['src_input']
        encoder_mask = batch['src_mask']
        assert (encoder_input.size(0) == 1), "batch size should be 1 for validation"

        model_out = greedy_decode(self.model, encoder_input, encoder_mask, self.tokenizer_src, self.tokenizer_tgt, self.config['seq_len'], self.device)

        source_text = batch['src_text'][0]
        target_text = batch['tgt_text'][0]
        model_out_text = self.tokenizer_tgt.decode(model_out.detach().cpu().numpy())
        