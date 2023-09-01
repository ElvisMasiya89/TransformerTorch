import torch
from torch.utils.data import Dataset


class BilingualDataset(Dataset):
    def __init__(self, dataset, tokenizer_source, tokenizer_target, source_lang, target_lang, seq_len):
        """
        Bilingual dataset class for training a sequence-to-sequence model.

        Args:
            dataset (list): List of dictionaries containing source and target translations.
            tokenizer_source: Tokenizer for source language.
            tokenizer_target: Tokenizer for target language.
            source_lang (str): Key for accessing source language in the dataset dictionary.
            target_lang (str): Key for accessing target language in the dataset dictionary.
            seq_len (int): Maximum sequence length for encoder and decoder inputs.
        """
        self.seq_len = seq_len
        self.dataset = dataset
        self.tokenizer_source = tokenizer_source
        self.tokenizer_target = tokenizer_target
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.sos_token = torch.tensor([tokenizer_source.token_to_id('[SOS]')], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer_source.token_to_id('[EOS]')], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_source.token_to_id('[PAD]')], dtype=torch.int64)

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.dataset)

    def __getitem__(self, index):
        """
        Retrieves a sample from the dataset and preprocesses it.

        Args:
            index (int): Index of the sample to retrieve.

        Returns:
            encoder_input (Tensor): Padded encoder input sequence.
            decoder_input (Tensor): Padded decoder input sequence.
            masks, labels, and text data.
        """
        source_target_pair = self.dataset[index]
        source_text = source_target_pair['translation'][self.source_lang]
        target_text = source_target_pair['translation'][self.target_lang]

        encoder_input_tokens = self.tokenizer_source.encode(source_text).ids
        decoder_input_tokens = self.tokenizer_target.encode(target_text).ids

        encoder_num_padding_tokens = self.seq_len - len(encoder_input_tokens) - 2
        decoder_num_padding_tokens = self.seq_len - len(decoder_input_tokens) - 1

        if encoder_num_padding_tokens < 0 or decoder_num_padding_tokens < 0:
            raise ValueError("Sentence is too long")

        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(encoder_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * encoder_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(decoder_input_tokens, dtype=torch.int64),
                torch.tensor([self.pad_token] * decoder_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        label = torch.cat(
            [
                torch.tensor(decoder_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * decoder_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {
            'encoder_input': encoder_input,  # (seq_len)
            'decoder_input': decoder_input,  # (seq_len)
            'encoder_mask': (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(),  # (1, 1, seq_len)
            'decoder_mask': (decoder_input != self.pad_token).unsqueeze(0).int() & causal_mask(decoder_input.size(0)),
            # (1, seq_len) & (1, seq_len, seq_len),
            'target_label': label,  # (seq_len)
            'source_text': source_text,
            'target_text': target_text,
        }


def causal_mask(size):
    """
        Creates a causal mask for the decoder.

        Args:
            size (int): Size of the mask.

        Returns:
            causal_mask (Tensor): Causal mask.
        """
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0
