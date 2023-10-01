from config import get_config
from train import train_model

def run_train():
    cfg = get_config()
    cfg['batch_size'] = 8
    cfg['num_epochs'] = 30

    train_model(cfg)
    
if __name__ == '__main__':
    run_train()

