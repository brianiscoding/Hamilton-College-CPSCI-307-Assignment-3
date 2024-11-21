import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
import sentencepiece as spm
import torchmetrics
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.nn.utils.rnn import pad_sequence
from torchmetrics.text import BLEUScore  # Import the correct BLEU score metric

# Load the tab-separated .txt file (assuming the file has no headers and columns are: English <tab> French)
df = pd.read_csv("eng_fra.txt", sep="\t", header=None, names=["english", "french"])

# Split the dataset into training, validation, and test sets (80% train, 10% val, 10% test)
train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)
train_data, val_data = train_test_split(train_data, test_size=0.1, random_state=42)

# Save the split data if necessary (optional)
train_data.to_csv("train_data.txt", sep="\t", index=False, header=False)
val_data.to_csv("val_data.txt", sep="\t", index=False, header=False)
test_data.to_csv("test_data.txt", sep="\t", index=False, header=False)


# Tokenization using SentencePiece for both languages
def train_sentencepiece_model(corpus, model_prefix, vocab_size=8000):
    # Save the corpus to a text file
    with open(f"{model_prefix}_corpus.txt", "w", encoding='utf-8') as f:
        for sentence in corpus:
            f.write(f"{sentence}\n")

    # Train the SentencePiece model
    spm.SentencePieceTrainer.train(
        input=f"{model_prefix}_corpus.txt",
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type="bpe",
    )


# Train SentencePiece models for both English and French datasets
train_sentencepiece_model(train_data["english"], "en_model")
train_sentencepiece_model(train_data["french"], "fr_model")

# Load the SentencePiece models
sp_en = spm.SentencePieceProcessor(model_file="en_model.model")
sp_fr = spm.SentencePieceProcessor(model_file="fr_model.model")


# Tokenize the sentences
def tokenize_sentence(sentence, sp_model):
    return sp_model.encode(sentence, out_type=str)


# Define a Dataset class to handle tokenized sentences
class TranslationDataset(Dataset):
    def __init__(self, english_sentences, french_sentences, sp_en, sp_fr):
        self.english_sentences = english_sentences
        self.french_sentences = french_sentences
        self.sp_en = sp_en
        self.sp_fr = sp_fr

    def __len__(self):
        return len(self.english_sentences)

    def __getitem__(self, idx):
        # Tokenize the sentences using SentencePiece models
        eng_tokens = self.sp_en.encode(self.english_sentences.iloc[idx], out_type=int)
        fr_tokens = self.sp_fr.encode(self.french_sentences.iloc[idx], out_type=int)

        return torch.tensor(eng_tokens), torch.tensor(fr_tokens)


def collate_fn(batch):
    # pad src and tgt sequences to same length
    src, tgt = zip(*batch)
    src_padded = pad_sequence(src, padding_value=0)
    tgt_padded = pad_sequence(tgt, padding_value=0)
    return src_padded, tgt_padded


# Create Datasets and DataLoaders
train_dataset = TranslationDataset(
    train_data["english"], train_data["french"], sp_en, sp_fr
)
val_dataset = TranslationDataset(val_data["english"], val_data["french"], sp_en, sp_fr)
test_dataset = TranslationDataset(
    test_data["english"], test_data["french"], sp_en, sp_fr
)

train_dataloader = DataLoader(train_dataset, batch_size=32, collate_fn=collate_fn)
val_dataloader = DataLoader(val_dataset, batch_size=32, collate_fn=collate_fn)
test_dataloader = DataLoader(test_dataset, batch_size=32, collate_fn=collate_fn)


# Define the Transformer Model
class Transformer(nn.Module):
    def __init__(
        self, input_dim, output_dim, emb_dim=256, n_heads=8, num_layers=6, dropout=0.1
    ):
        super(Transformer, self).__init__()

        # Embeddings for input and output languages
        self.src_embedding = nn.Embedding(input_dim, emb_dim)
        self.tgt_embedding = nn.Embedding(output_dim, emb_dim)

        # Positional encoding layer
        self.positional_encoding = nn.Parameter(
            torch.zeros(1, 1000, emb_size)
        )  # Ensure this size is sufficient

        # Transformer Encoder and Decoder
        self.transformer = nn.Transformer(
            d_model=emb_dim,
            nhead=n_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dropout=dropout,
        )

        # Output layer
        self.fc_out = nn.Linear(emb_dim, output_dim)

        # Activation function
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, src, tgt):

        # Add positional encoding to input and target
        src = self.src_embedding(src) + self.positional_encoding[:, : src.size(1), :]
        tgt = self.tgt_embedding(tgt) + self.positional_encoding[:, : tgt.size(1), :]

        # Pass through Transformer
        output = self.transformer(src, tgt)

        # Pass through output layer
        output = self.fc_out(output)
        return output

# Define the TranslationModel
class TranslationModel(pl.LightningModule):
    def __init__(
        self, input_dim, output_dim, sp_en, sp_fr, emb_dim=256, n_heads=8, num_layers=6, dropout=0.1
    ):
        super(TranslationModel, self).__init__()
        self.model = Transformer(
            input_dim, output_dim, emb_dim, n_heads, num_layers, dropout
        )
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding tokens
        self.sp_en = sp_en  # Store the SentencePiece model for English
        self.sp_fr = sp_fr  # Store the SentencePiece model for French
        # BLEU score metric (ensure you have the right metric for translation)
        self.bleu_score = BLEUScore(n_gram=4)

    def forward(self, src, tgt=None):
        src = self.model.src_embedding(src) + self.model.positional_encoding[:, :src.size(1), :]
        
        if tgt is not None:
            tgt = self.model.tgt_embedding(tgt) + self.model.positional_encoding[:, :tgt.size(1), :]
            output = self.model.transformer(src, tgt)
        else:
            output = self.model.transformer.encoder(src)

        output = self.model.fc_out(output)
        return output

    def training_step(self, batch, batch_idx):
        src, tgt = batch
        tgt_input = tgt[:-1, :]
        tgt_output = tgt[1:, :]

        output = self(src, tgt_input)

        loss = self.loss_fn(output.view(-1, output.size(-1)), tgt_output.view(-1))
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        src, tgt = batch
        tgt_input = tgt[:-1, :]
        tgt_output = tgt[1:, :]

        output = self(src, tgt_input)

        val_loss = self.loss_fn(output.view(-1, output.size(-1)), tgt_output.view(-1))
        self.log("val_loss", val_loss, on_epoch=True, prog_bar=True)
        
        # BLEU score calculation for validation
        bleu_score = self.calculate_bleu(output, tgt)
        self.log("val_bleu", bleu_score, on_epoch=True, prog_bar=True)
        return val_loss

    def calculate_bleu(self, preds, targets):
        # Apply argmax to get the predicted tokens (ignoring the vocab dimension)
        preds = preds.argmax(dim=-1)  # Shape will be [seq_len, batch_size]

        # Swap dimensions so that preds and targets match the expected format [batch_size, seq_len]
        preds = preds.permute(1, 0)  # Now it's [batch_size, seq_len]
        targets = targets.permute(1, 0)  # Ensure targets have the same format

        # Convert to list of tokenized sequences
        preds = preds.tolist()  # Each element will be a list of integers (tokens)
        targets = targets.tolist()  # Each element will be a list of integers (tokens)

        # Detokenize the predictions and targets using SentencePiece models
        detok_preds = [self.sp_fr.decode(p) for p in preds]  # Detokenize predictions
        detok_targets = [self.sp_fr.decode(t) for t in targets]  # Detokenize targets

        # Now pass detokenized sequences (strings) to the BLEUScore metric
        bleu_score = self.bleu_score(detok_preds, detok_targets)
        return bleu_score

    def test_step(self, batch, batch_idx):
        src, tgt = batch

        tgt_input = torch.zeros((1, src.size(1)), dtype=torch.long, device=self.device)

        for _ in range(50):  # Max decode length
            output = self(src, tgt_input)
            next_token = output[-1].argmax(dim=-1, keepdim=True)
            tgt_input = torch.cat([tgt_input, next_token], dim=0)

            if torch.all(next_token == 1):
                break

        bleu_score = self.calculate_bleu(tgt_input, tgt)
        self.log("test_bleu", bleu_score, prog_bar=True)
        return {"test_bleu": bleu_score}

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.model.parameters(), lr=1e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.7)
        return [optimizer], [scheduler]

# PyTorch Lightning Trainer setup
wandb_logger = WandbLogger(project="eng2fre-translation")

# Early stopping callback
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

# Model parameters
vocab_size_src = 8000  # Replace with actual vocab size for English
vocab_size_tgt = 8000  # Replace with actual vocab size for French
emb_size = 512  # Embedding size, make sure it's divisible by num_heads
num_heads = 8  # Number of attention heads, should divide emb_size evenly
hidden_size = 2048
num_layers = 6  # Number of layers

# Instantiate the model
model = TranslationModel(
    vocab_size_src, vocab_size_tgt, sp_en, sp_fr, emb_size, num_heads, num_layers
)
# ------------------ TRAINING WITH PyTorch Lightning ------------------ #

trainer = pl.Trainer(
    logger=wandb_logger,  # Use W&B logger for experiment tracking
    max_epochs=10,  # Set the number of epochs
    accelerator = "gpu" if torch.cuda.is_available() else "cpu",  # Specify the use of GPU 
    precision=16,  # Use mixed precision if available
    callbacks=[checkpoint_callback, early_stopping],  # Implement checkpointing and early stopping
)

# Train the model
trainer.fit(model, train_dataloader, val_dataloader)

# ------------------ TESTING WITH PyTorch Lightning ------------------ #

# trainer.test(model, test_dataloader)