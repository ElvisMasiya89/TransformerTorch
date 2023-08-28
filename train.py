import torch
import torch.nn as nn
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from pathlib import Path


def get_all_sentences(dataset, lang):
    for sentence in dataset:
        yield sentence['translation'][lang]


def build_tokenizer(config, dataset, lang):
    # eg config['tokenizer_file'] = '../tokenizer/tokenizer_en.json'
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_tokenizer(get_all_sentences(dataset, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))

    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))

    return tokenizer


def get_dataset(config):
    dataset = load_dataset(config['opus_books'], f'{config["lang_source"]}-{config["lang_target"]}', split='train')

    # Build tokenizers
    tokenizer_source = build_tokenizer(config, dataset, config['lang_source'])
    tokenizer_target = build_tokenizer(config, dataset, config['lang_target'])

    # Keep 90% for training and 10% for validation
    training_dataset_size = int(len(dataset) * 0.9)
    validation_dataset_size = int(len(dataset) * 0.1)
    training_dataset, validation_dataset = torch.utils.data.random_split(dataset, [training_dataset_size, validation_dataset_size])



