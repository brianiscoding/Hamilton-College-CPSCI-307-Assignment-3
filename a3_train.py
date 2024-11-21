import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.nn.utils.rnn import pad_sequence
import math
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import sentencepiece as spm
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

# ------------------------------
# Preprocessing: Tokenization
# ------------------------------

def preprocess_and_split(input_file, vocab_size=8000, test_size=0.2, random_state=42):
    # Read the tab-separated text file
    with open(input_file, "r") as f:
        data = f.readlines()

    # Split data into English and French sentences
    english_sentences = []
    french_sentences = []
    for line in data:
        english, french = line.strip().split("\t")
        english_sentences.append(english)
        french_sentences.append(french)

    # Train SentencePiece tokenizers for English and French
    def train_tokenizer(sentences, model_prefix):
        temp_file = f"{model_prefix}_temp.txt"
        with open(temp_file, "w", encoding='utf-8') as f:
            f.write("\n".join(sentences))
        spm.SentencePieceTrainer.train(
            input=temp_file, model_prefix=model_prefix, vocab_size=vocab_size, character_coverage=1.0
        )

    # Train tokenizers
    train_tokenizer(english_sentences, "english_bpe")
    train_tokenizer(french_sentences, "french_bpe")

    # Load tokenizers
    english_tokenizer = spm.SentencePieceProcessor(model_file="english_bpe.model")
    french_tokenizer = spm.SentencePieceProcessor(model_file="french_bpe.model")

    # Tokenize English and French sentences
    english_tokens = [english_tokenizer.encode(x, out_type=int) for x in english_sentences]
    french_tokens = [french_tokenizer.encode(x, out_type=int) for x in french_sentences]

    # Split data into training and validation sets
    train_data, val_data = train_test_split(list(zip(english_tokens, french_tokens)), test_size=test_size, random_state=random_state)

    return train_data, val_data, english_tokenizer, french_tokenizer


# ------------------------------
# Dataset and DataModule
# ------------------------------

# Define a Dataset class to handle tokenized sentences
class TranslationDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src, tgt = self.data[idx]
        src = torch.tensor(src, dtype=torch.long)
        tgt = torch.tensor(tgt, dtype=torch.long)
        return src, tgt

# Define a collate function to pad sequences dynamically
def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)

    # Pad sequences dynamically for each batch
    src_batch = pad_sequence(src_batch, batch_first=True, padding_value=0)
    tgt_batch = pad_sequence(tgt_batch, batch_first=True, padding_value=0)

    return src_batch, tgt_batch

# Define a DataModule class
class TranslationDataModule(pl.LightningDataModule):
    def __init__(self, train_data, val_data, batch_size=32):
        super().__init__()
        self.train_data = train_data
        self.val_data = val_data
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.train_dataset = TranslationDataset(self.train_data)
        self.val_dataset = TranslationDataset(self.val_data)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, collate_fn=collate_fn)


# ------------------------------
# Seq2Seq Transformer Model
# ------------------------------

# Define a Positional Encoding class
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

# Define a Decoder class
class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers, norm=nn.LayerNorm(d_model))

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None):
        tgt = self.embedding(tgt) * math.sqrt(self.embedding.embedding_dim)
        tgt = self.pos_encoder(tgt.transpose(0, 1))  # Transformer expects (seq_len, batch_size, d_model)
        output = self.decoder(
            tgt, 
            memory.transpose(0, 1),  # Transpose memory to (seq_len, batch_size, d_model)
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )
        return output.transpose(0, 1)  # Return to (batch_size, tgt_seq_len, d_model)

# Define an Encoder class
class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, dim_feedforward=2048, dropout=0.1):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
    
    def forward(self, src, src_key_padding_mask=None):        
        src = self.embedding(src) * math.sqrt(self.embedding.embedding_dim)
        src = self.pos_encoder(src.transpose(0, 1))  # Transformer expects (seq_len, batch_size, d_model)
        output = self.encoder(
            src, 
            src_key_padding_mask=src_key_padding_mask
        )
        return output.transpose(0, 1)  # Return to (batch_size, seq_len, d_model)

# Define a Seq2SeqModel
class Seq2SeqModel(pl.LightningModule):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers,
                 dim_feedforward, lr, max_len=5000, dropout=0.1):
        super().__init__()
        self.save_hyperparameters()

        self.encoder = Encoder(src_vocab_size, d_model, nhead, num_encoder_layers, dim_feedforward, dropout)
        self.decoder = Decoder(tgt_vocab_size, d_model, nhead, num_decoder_layers, dim_feedforward, dropout)
        self.output_layer = nn.Linear(d_model, tgt_vocab_size)
        self.softmax = nn.Softmax(dim=-1)
        self.lr = lr
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, src_padding_mask=None, tgt_padding_mask=None):
        memory = self.encoder(src, src_padding_mask)
        output = self.decoder(
            tgt, memory,
            tgt_mask=tgt_mask,
            memory_mask=None,  # Typically not used
            tgt_key_padding_mask=tgt_padding_mask
        )
        output = self.output_layer(output)
        return output

    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool()
        return mask

    def training_step(self, batch, batch_idx):
        src, tgt = batch
        tgt_input = tgt[:, :-1]  # Input to decoder
        tgt_output = tgt[:, 1:]  # Target for loss

        # Generate masks
        src_mask = None
        tgt_mask = self.generate_square_subsequent_mask(tgt_input.size(1))
        train_loss = self.forward(src, tgt_input, src_mask, tgt_mask)
        train_loss = self.criterion(train_loss.view(-1, train_loss.size(-1)), tgt_output.contiguous().view(-1))

        self.log("train_loss", train_loss, on_epoch=True, prog_bar=True)

        return train_loss

    def validation_step(self, batch, batch_idx):
        src, tgt = batch
        tgt_input = tgt[:, :-1]  # Input to decoder
        tgt_output = tgt[:, 1:]  # Target for loss

        # Generate masks
        src_mask = None
        tgt_mask = self.generate_square_subsequent_mask(tgt_input.size(1))
        val_loss = self.forward(src, tgt_input, src_mask, tgt_mask)
        val_loss = self.criterion(val_loss.view(-1, val_loss.size(-1)), tgt_output.contiguous().view(-1))

        # BLEU Score Computation
        predictions = self.decode(src, max_len=tgt.size(1))
        predictions = predictions.cpu().tolist()
        references = tgt.cpu().tolist()

        # Convert to list of sentences (tokens), where each sentence is a list of tokens
        # BLEU expects references and predictions to be in this form
        pred_sentences = [[int(x) for x in pred] for pred in predictions]
        ref_sentences = [[int(x) for x in ref] for ref in references]

        # Compute BLEU score
        bleu_score = 0.0
        for pred, ref in zip(pred_sentences, ref_sentences):
            bleu_score += sentence_bleu([ref], pred, smoothing_function=SmoothingFunction().method1)

        bleu_score /= len(pred_sentences)  # Average BLEU score over the batch

        self.log("val_loss", val_loss, prog_bar=True)
        self.log("val_bleu", bleu_score, prog_bar=True)

        return val_loss
    
    def decode(self, src, max_len=50, start_token_id=2, end_token_id=3):
        memory = self.encoder(src)
        ys = torch.ones(src.size(0), 1).fill_(start_token_id).type_as(src)

        for i in range(max_len - 1):
            tgt_mask = self.generate_square_subsequent_mask(ys.size(1)).to(self.device)
            output = self.decoder(ys, memory, tgt_mask)
            prob = self.output_layer(output[:, -1, :])
            prob = self.softmax(prob)
            _, next_word = torch.max(prob, dim=1)
            ys = torch.cat([ys, next_word.unsqueeze(1)], dim=1)

            if torch.all(next_word == end_token_id):
                break
        return ys

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


# ------------------------------
# Main Training Code
# ------------------------------

if __name__ == "__main__":
    # Define hyperparameters
    input_file = "eng_fra.txt"
    batch_size = 32
    vocab_size = 8000
    lr = 1e-4
    d_model = 512
    nhead = 8
    num_encoder_layers = 6
    num_decoder_layers = 6
    dim_feedforward = 2048

    # Preprocess and split data
    train_data, val_data, english_tokenizer, french_tokenizer = preprocess_and_split(input_file, vocab_size)

    # Initialize model and data module
    model = Seq2SeqModel(
        len(english_tokenizer), len(french_tokenizer), d_model, nhead,
        num_encoder_layers, num_decoder_layers, dim_feedforward, lr
    )
    data_module = TranslationDataModule(train_data, val_data, batch_size)

    # Initialize W&B logger
    wandb_logger = WandbLogger(project="eng2fre-translation")

    # Add early stopping and checkpointing
    early_stopping = EarlyStopping(
        monitor="val_loss",  # Metric to monitor
        patience=3,          # Number of epochs with no improvement to wait before stopping
        verbose=True,        # Print a message when stopping
        mode="min"           # "min" because we're minimizing validation loss
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",  # Metric to monitor
        save_top_k=1,        # Save only the best model
        mode="min",          # "min" for loss, "max" for metrics like accuracy
        save_weights_only=False,  # Save full model (weights + architecture)
        dirpath="checkpoints",    # Directory to save checkpoints
        filename="model-{epoch:02d}-{val_loss:.2f}"  # Naming convention
    )

    # Train the model
    device = "gpu" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    trainer = pl.Trainer(
        max_epochs=20,
        accelerator=device,
        logger=wandb_logger,
        callbacks=[early_stopping, checkpoint_callback],
        gradient_clip_val=1.0)
    
    trainer.fit(model, data_module)

    print("Training completed successfully!")