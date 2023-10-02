import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from config import get_config
from train_lightning import CustomLightningModule, get_model, get_ds

def run_train():
    cfg = get_config()
    cfg['batch_size'] = 8
    cfg['num_epochs'] = 10
    cfg['dropout'] = 0.2
    cfg['d_ff'] = 2048
    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(cfg)
    model = get_model(cfg, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size())
    lightning_model = CustomLightningModule(cfg, model, tokenizer_src, tokenizer_tgt)
    logger = WandbLogger(project="transformer_optimization", log_model="all")
    trainer = pl.Trainer(accelerator='gpu', max_epochs=cfg['num_epochs'], logger=logger)
    trainer.fit(lightning_model, train_dataloader, val_dataloader)
