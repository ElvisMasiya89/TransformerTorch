import torch
import math

from model import InputEmbeddings, build_transformer


def test_transformer():
    num_layers = 6
    d_model = 512
    num_heads = 8
    dff = 2048
    dropout = 0.1
    source_vocab_size = 10000  # Adjust this based on your dataset
    target_vocab_size = 10000  # Adjust this based on your dataset
    max_seq_len = 274  # Adjust this based on your use case

    # Build the Transformer model
    transformer = build_transformer(num_layers, d_model, num_heads, dff, dropout,
                                    source_vocab_size, target_vocab_size, max_seq_len)

    # Create random input tensors for source and target
    source_input = torch.randint(low=0, high=source_vocab_size, size=(1, max_seq_len))
    target_input = torch.randint(low=0, high=target_vocab_size, size=(1, max_seq_len))

    # Mask for padding
    source_mask = (source_input != 0)

    # Pass input through the Transformer
    encoder_output = transformer.encode(source_input, source_mask)
    decoder_output = transformer.decode(target_input, encoder_output, source_mask, source_mask)
    output_logits = transformer.project(decoder_output)

    print("Source input shape:", source_input.shape)
    print("Source mask shape:", source_mask.shape)
    print("Target input shape:", target_input.shape)
    print("Encoder output shape:", encoder_output.shape)
    print("Decoder output shape:", decoder_output.shape)
    print("Output logits shape:", output_logits.shape)


if __name__ == "__main__":
    test_transformer()
