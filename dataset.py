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
        self.dataset = dataset
        self.tokenizer_source = tokenizer_source
        self.tokenizer_target = tokenizer_target
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.seq_len = seq_len

        self.sos_token = torch.Tensor([tokenizer_source.token_to_id(['[SOS]'])], dtype=torch.int64)
        self.eos_token = torch.Tensor([tokenizer_source.token_to_id(['[EOS]'])], dtype=torch.int64)
        self.pad_token = torch.Tensor([tokenizer_source.token_to_id(['[PAD]'])], dtype=torch.int64)

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.dataset)

    def __getitem__(self, index):
        """
            Retrieves a sample from the dataset and preprocesses it.

            Args:
                index (int): Index of the sample to retrieve.

            Returns:
                encoder_input_text (Tensor): Padded or truncated encoder input sequence.
                decoder_input_text (Tensor): Padded or truncated decoder input sequence.
            """

        source_target_pair = self.dataset[index]
        source_text = source_target_pair['translation'][self.source_lang]
        target_text = source_target_pair['translation'][self.target_lang]

        encoder_input_tokens = self.tokenizer_source.encode(source_text).ids
        decoder_input_tokens = self.tokenizer_target.encode(target_text).ids

        # Add SOS tokens to decoder input
        decoder_input = [self.sos_token.item()] + decoder_input_tokens

        # Add SOS and EOS tokens to encoder input
        encoder_input = [self.sos_token.item()] + encoder_input_tokens + [self.eos_token.item()]

        # Generate target labels by excluding the last token (EOS token) from the decoder input
        target_label = decoder_input[:-1]

        # Pad or truncate encoder and decoder input sequences
        encoder_input = self._pad_or_truncate(encoder_input, self.seq_len)
        decoder_input = self._pad_or_truncate(decoder_input, self.seq_len)
        target_label = self._pad_or_truncate(target_label, self.seq_len)

        # Convert to tensors
        encoder_input = torch.tensor(encoder_input, dtype=torch.int64)
        decoder_input = torch.tensor(decoder_input, dtype=torch.int64)
        target_label = torch.tensor(target_label, dtype=torch.int64)

        # Return data as a dictionary
        return {
            'encoder_input': encoder_input,  # (Seq_Len)
            'decoder_input': decoder_input,
            'encoder_mask': (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(),  # (1,1,Seq_Len)
            'decoder_mask': (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int()
                            & self.casual_mask(decoder_input.size(0)),  # (1,1,Seq_Len) & (1,Seq_Len , Seq_Len)
            'target_label': target_label,
            'source_text': source_text,
            'target_text': target_text
        }

    def _pad_or_truncate(self, sequence, seq_len):
        """
        Pads or truncates the sequence to the specified length.

        Args:
            sequence (list): Sequence to be padded or truncated.
            seq_len (int): Desired sequence length.

        Returns:
            padded_sequence (list): Padded or truncated sequence.
        """
        if len(sequence) < seq_len:
            sequence += [self.pad_token.item()] * (seq_len - len(sequence))
        elif len(sequence) > seq_len:
            sequence = sequence[:seq_len]
        return sequence

    def casual_mask(self, seq_len):
        """
        Creates a causal mask for the decoder.

        Args:
            seq_len (int): Sequence length.

        Returns:
            causal_mask (Tensor): Causal mask.
        """
        causal_mask = torch.triu(torch.ones(1, seq_len, seq_len), diagonal=1).type(torch.int)
        return causal_mask == 0
