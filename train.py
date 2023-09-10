from pathlib import Path
import torch
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import get_weights_file_path, get_config
from dataset import BilingualDataset, causal_mask
from model import build_transformer


def greedy_decode(model, source, encoder_mask, tokenizer_source, tokenizer_target, max_length, device):
    sos_idx = tokenizer_target.token_to_id('[SOS]')
    eos_idx = tokenizer_target.token_to_id('[EOS]')

    # Pre-compute the encoder output and reuse it for every token we get from the decoder
    encoder_output = model.encode(source, encoder_mask)
    # Initialize the decoder input with sos token
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(encoder_mask).to(device)

    while True:
        if decoder_input.size(1) == max_length:
            break

        # Build the mask for the target ( decoder input )
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(encoder_mask).to(device)

        # Calculate the output of the decoder
        decoder_output = model.decode(decoder_input, encoder_output, encoder_mask,
                                      decoder_mask)

        # Get the next token
        probabilities = model.project(decoder_output[:, -1])

        # Select the token with the max probability (because it is greedy search)
        _, next_word = torch.max(probabilities, dim=1)

        decoder_input = torch.cat([decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)],
                                  dim=1)

        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)


def run_validation(model, validation_dataset, tokenizer_source, tokenizer_target, max_length, device, print_msg,
                   global_state, writer, num_examples=2):
    model.eval()
    count = 0

    # source_texts = []
    # expected = []
    # predicted = []

    # Size of the control window( just us a default value)
    console_width = 80
    with torch.no_grad():
        for batch in validation_dataset:
            count += 1
            encoder_input = batch['encoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)

            assert encoder_input.size(0) == 1, " Batch size must be 1 for validation"

            model_output = greedy_decode(model, encoder_input, encoder_mask, tokenizer_source, tokenizer_target,
                                         max_length,
                                         device)

            source_text = batch['source_text'][0]
            target_text = batch['target_text'][0]

            model_output_text = tokenizer_target.decode(model_output.detach().cpu().numpy())

            # source_texts.append(source_text)
            # expected.append(target_text)
            # predicted.append(model_output_text)

            # Print to the console

            print_msg('-' * console_width)
            print_msg(f'SOURCE{source_text}')
            print_msg(f'TARGET{target_text}')
            print_msg(f'PREDICT{model_output_text}')

            if count == num_examples:
                break


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
        tokenizer.train_from_iterator(get_all_sentences(dataset, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))

    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))

    return tokenizer


def get_dataset(config):
    dataset = load_dataset('opus_books', f'{config["lang_source"]}-{config["lang_target"]}', split='train')

    # Build tokenizers
    tokenizer_source = build_tokenizer(config, dataset, config['lang_source'])
    tokenizer_target = build_tokenizer(config, dataset, config['lang_target'])

    # Keep 90% for training and 10% for validation
    training_dataset_size = int(len(dataset) * 0.9)
    validation_dataset_size = len(dataset) - training_dataset_size
    training_dataset_raw, validation_dataset_raw = torch.utils.data.random_split(dataset, [training_dataset_size,
                                                                                           validation_dataset_size])

    # Calculate the maximum sequence lengths for source and target languages
    max_len_source = 0
    max_len_target = 0
    for item in training_dataset_raw:
        source_text = item['translation'][config['lang_source']]
        target_text = item['translation'][config['lang_target']]
        max_len_source = max(max_len_source, len(tokenizer_source.encode(source_text).ids))
        max_len_target = max(max_len_target, len(tokenizer_target.encode(target_text).ids))

    training_dataset = BilingualDataset(training_dataset_raw, tokenizer_source, tokenizer_target,
                                        config['lang_source'],
                                        config['lang_target'], max_len_target)

    validation_dataset = BilingualDataset(validation_dataset_raw, tokenizer_source, tokenizer_target,
                                          config['lang_source'],
                                          config['lang_target'], max_len_target)

    # Set the maximum sequence lengths in the configuration
    config['seq_len'] = max_len_source
    config['max_seq_len'] = max_len_source

    print(f" Max length of source text: {max_len_source}")
    print(f" Max length of target text: {max_len_target}")

    # Create data loaders for training and validation datasets
    training_dataloader = torch.utils.data.DataLoader(training_dataset, batch_size=config['batch_size'], shuffle=True)
    validation_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size=1, shuffle=False)

    return training_dataloader, validation_dataloader, tokenizer_source, tokenizer_target


def get_model(config, vocab_source_length, vocab_target_length):
    """
    Builds and returns a transformer model.

    Args:
        config (dict): Configuration settings.
        vocab_source_length (int): Vocabulary size for source language.
        vocab_target_length (int): Vocabulary size for target language.

    Returns:
        model (nn.Module): Transformer model.
    """
    # Extract model configuration parameters from config
    num_layers = config['num_layers']
    d_model = config['d_model']
    num_heads = config['num_heads']
    dff = config['dff']
    dropout = config['dropout']
    max_seq_len = config['seq_len']

    # Build the transformer model
    model = build_transformer(num_layers, d_model, num_heads, dff, dropout, vocab_source_length,
                              vocab_target_length, max_seq_len)

    return model


def train_model(config):
    # Define the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)

    # Get training data, validation data, source tokenizer, and target tokenizer
    training_dataloader, validation_dataloader, tokenizer_source, tokenizer_target = get_dataset(config)

    # Get vocabulary sizes for source and target languages from tokenizers
    vocab_source_length = tokenizer_source.get_vocab_size()
    vocab_target_length = tokenizer_target.get_vocab_size()

    # Build the transformer model
    model = get_model(config, vocab_source_length, vocab_target_length)

    # Tensorboard
    writer = SummaryWriter(config['experiment_name'])

    # Define loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss(ignore_index=tokenizer_target.token_to_id("[PAD]"),
                                          label_smoothing=0.1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], eps=1e-9)

    initial_epoch = 0
    global_step = 0

    if config['preload']:
        model_filename = get_weights_file_path(config, config['preload'])
        print(" Pre-Loading model", model_filename)
        state = torch.load(model_filename)
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']

    # Training loop
    num_epochs = config['num_epochs']
    for epoch in range(initial_epoch, num_epochs):
        model.train()
        batch_iterator = tqdm(training_dataloader, desc=f'Processing epoch {epoch:02d}')
        total_loss = 0

        for batch in training_dataloader:
            model.train()
            encoder_input = batch['encoder_input'].to(device)  # (Batch , Seq_Len)
            decoder_input = batch['decoder_input'].to(device)  # (Batch , Seq_Len)

            encoder_mask = batch['encoder_mask'].to(device)  # (Batch ,1 ,1 ,Seq_Len)
            decoder_mask = batch['decoder_mask'].to(device)  # (Batch ,1 ,Seq_len ,Seq_Len)

            print("Encoder Input Shape:", encoder_input.shape)
            print("Decoder Input Shape:", decoder_input.shape)
            print("Encoder Mask Shape:", encoder_mask.shape)
            print("Decoder Mask Shape:", decoder_mask.shape)

            # Run the tensors through the transformer

            encoder_output = model.encode(encoder_input, encoder_mask)  # (Batch , Seq_Len, d_model)
            decoder_output = model.decode(decoder_input, encoder_output, encoder_mask,
                                          decoder_mask)  # (Batch  Seq_Len, d_model)
            projection_output = model.project(decoder_output)  # (Batch, Seq_Len, target_vocab_size)

            target_label = batch['target_label'].to(device)  # (Batch, Seq_Len)

            # Calculate the loss
            # Flatten the projection_output and target_label tensors for the CrossEntropyLoss
            # Projection output shape after view: (Batch * Seq_Len, target_vocab_size)
            # Target label shape after view: (Batch * Seq_Len)
            loss = criterion(projection_output.view(-1, projection_output.shape[-1]), target_label.view(-1))
            batch_iterator.set_postfix({f"loss": f"{loss.item():6.3f}"})
            total_loss += loss.item()

            # Backpropagation and optimization step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()



            # Increment the global step count
            global_step += 1

        run_validation(model, validation_dataloader, tokenizer_source, tokenizer_target, config['seq_len'],
                       device, lambda msg: batch_iterator.write(msg), global_step, writer)

        # Calculate the average loss for the epoch
        avg_loss = total_loss / len(training_dataloader)

        # Log the loss to Tensorboard
        writer.add_scalar('Training Loss', avg_loss, global_step)
        writer.flush()

        # Print epoch info
        print(f"Epoch [{epoch + 1}/{num_epochs}] - Loss: {avg_loss:.4f}")

        # Save the trained model

        model_filename = get_weights_file_path(config, f'{epoch:02d}')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
            'global_step': global_step
        }, model_filename)


if __name__ == "__main__":
    # Load the configuration
    config = get_config()
    # Start training
    train_model(config)
