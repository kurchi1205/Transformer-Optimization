from pathlib import Path


def get_config():
    return {
        "batch_size": 2048,
        "num_epochs": 20,
        "lr": 1e-4,
        "seq_len": 350,
        "d_model": 512,
        "N": 6,
        "head": 8, 
        "lang_src": "en",
        "lang_tgt": "fr",
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "preload": None,
        "tokanizer_file": "tokenizer_{0}.json",
        "experiment_name": "transformer_opt"
    }


def get_weight_file_path(config, epoch: str):
    model_folder = config["model_folder"]
    model_basename = config["model_basename"]
    model_filename = f"{model_basename}{epoch}.pth"
    return str(Path(".")/model_folder/model_filename)



