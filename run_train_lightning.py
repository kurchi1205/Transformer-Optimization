import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.plugins.precision import MixedPrecisionPlugin
import torch

from config import get_config
from train_lightning import CustomLightningModule, get_model, get_ds

def run_train():
    cfg = get_config()
    cfg['batch_size'] = 64
    cfg['num_epochs'] = 20
    cfg['dropout'] = 0.2
    cfg['d_ff'] = 2048
    cfg['clean_data'] = True
    cfg['use_mixed_precision'] = False
    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(cfg)
    model = get_model(cfg, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size())
    lightning_model = CustomLightningModule(cfg, model, tokenizer_src, tokenizer_tgt)
    logger = WandbLogger(project="transformer_optimization", log_model="all")
    checkpoint = ModelCheckpoint(
        dirpath=cfg["model_folder"],
        filename='tmodel_{epoch}',
        every_n_epochs=1
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    plugin = MixedPrecisionPlugin('16-mixed', device=device)
    
    plugins = []
    if cfg['use_mixed_precision']:
        plugins = plugin
        
    trainer = pl.Trainer(accelerator='gpu', max_epochs=cfg['num_epochs'], logger=logger, callbacks=[checkpoint], limit_val_batches=10, plugins=plugins)
    
    trainer.fit(lightning_model, train_dataloader, val_dataloaders=val_dataloader)

if __name__ == "__main__":
    run_train()