from pathlib import Path


def get_config():
    return {
        'lang_source': 'en',
        'lang_target': 'it',
        'tokenizer_file': 'tokenizer_{0}.json',  # Provide the path to your tokenizer directory
        'batch_size': 1,
        'num_layers': 4,
        'd_model': 512,
        'num_heads': 8,
        'dff': 1024,
        'dropout': 0.1,
        'learning_rate': 10 ** -4,
        'num_epochs': 20,
        'model_folder': "weights",
        'model_basename': "transformer_model_",
        'preload': None,
        'experiment_name': "runs/transformer_model"

    }


def get_weights_file_path(config, epoch):
    model_folder = config['model_folder']
    model_basename = config['model_basename']
    model_filename = f"{model_basename}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)
