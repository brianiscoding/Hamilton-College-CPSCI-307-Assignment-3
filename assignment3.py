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
from torch.nn.utils.rnn import pad_sequence
from torchmetrics.text import BLEUScore  # Import the correct BLEU score metric

# Load the tab-separated .txt file (assuming the file has no headers and columns are: English <tab> French)
df = pd.read_csv('eng_fra.txt', sep='\t', header=None, names=['english', 'french'])

# Split the dataset into training, validation, and test sets (80% train, 10% val, 10% test)
train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)
train_data, val_data = train_test_split(train_data, test_size=0.1, random_state=42)

# Save the split data if necessary (optional)
train_data.to_csv('train_data.txt', sep='\t', index=False, header=False)
val_data.to_csv('val_data.txt', sep='\t', index=False, header=False)
test_data.to_csv('test_data.txt', sep='\t', index=False, header=False)

# Tokenization using SentencePiece for both languages
def train_sentencepiece_model(corpus, model_prefix, vocab_size=8000):
    # Save the corpus to a text file
    with open(f'{model_prefix}_corpus.txt', 'w') as f:
        for sentence in corpus:
            f.write(f"{sentence}\n")
    
    # Train the SentencePiece model
    spm.SentencePieceTrainer.train(input=f'{model_prefix}_corpus.txt', model_prefix=model_prefix, vocab_size=vocab_size, model_type='bpe')

# Train SentencePiece models for both English and French datasets
train_sentencepiece_model(train_data['english'], 'en_model')
train_sentencepiece_model(train_data['french'], 'fr_model')

# Load the SentencePiece models
sp_en = spm.SentencePieceProcessor(model_file='en_model.model')
sp_fr = spm.SentencePieceProcessor(model_file='fr_model.model')

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
train_dataset = TranslationDataset(train_data['english'], train_data['french'], sp_en, sp_fr)
val_dataset = TranslationDataset(val_data['english'], val_data['french'], sp_en, sp_fr)
test_dataset = TranslationDataset(test_data['english'], test_data['french'], sp_en, sp_fr)

train_dataloader = DataLoader(train_dataset, batch_size=32, collate_fn=collate_fn)
val_dataloader = DataLoader(val_dataset, batch_size=32, collate_fn=collate_fn)
test_dataloader = DataLoader(test_dataset, batch_size=32, collate_fn=collate_fn)

# Define the Transformer Model
class Transformer(nn.Module):
    def __init__(self, input_dim, output_dim, emb_dim=256, n_heads=8, num_layers=6, dropout=0.1):
        super(Transformer, self).__init__()
        
        # Embeddings for input and output languages
        self.src_embedding = nn.Embedding(input_dim, emb_dim)
        self.tgt_embedding = nn.Embedding(output_dim, emb_dim)
        
        # Positional encoding layer
        self.positional_encoding = nn.Parameter(torch.zeros(1, 1000, emb_size))  # Ensure this size is sufficient
        
        # Transformer Encoder and Decoder
        self.transformer = nn.Transformer(d_model=emb_dim, nhead=n_heads, num_encoder_layers=num_layers, 
                                          num_decoder_layers=num_layers, dropout=dropout)
        
        # Output layer
        self.fc_out = nn.Linear(emb_dim, output_dim)
        
        # Activation function
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, src, tgt):

        # Add positional encoding to input and target
        src = self.src_embedding(src) + self.positional_encoding[:, :src.size(1), :]
        tgt = self.tgt_embedding(tgt) + self.positional_encoding[:, :tgt.size(1), :]
        
        # Pass through Transformer
        output = self.transformer(src, tgt)
        
        # Pass through output layer
        output = self.fc_out(output)
        return output

# Define the Lightning Module for training
class TranslationModel(pl.LightningModule):
    def __init__(self, input_dim, output_dim, emb_dim=256, n_heads=8, num_layers=6, dropout=0.1):
        super(TranslationModel, self).__init__()
        self.model = Transformer(input_dim, output_dim, emb_dim, n_heads, num_layers, dropout)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding tokens
        # BLEU score metric (ensure you have the right metric for translation)
        self.bleu_score = torchmetrics.CohenKappa(num_classes=4, task="multiclass")

    def forward(self, src, tgt):
        return self.model(src, tgt)

    def training_step(self, batch, batch_idx):
        src, tgt = batch
        tgt_input = tgt[:-1, :]
        tgt_output = tgt[1:, :]
        
        # Forward pass
        output = self(src, tgt_input)
        
        # Compute the loss
        loss = self.loss_fn(output.view(-1, output.size(-1)), tgt_output.view(-1))
        return loss

    def validation_step(self, batch, batch_idx):
        src, tgt = batch
        tgt_input = tgt[:-1, :]
        tgt_output = tgt[1:, :]
        
        # Forward pass
        output = self(src, tgt_input)
        
        # Compute the loss
        loss = self.loss_fn(output.view(-1, output.size(-1)), tgt_output.view(-1))
        return loss

    def test_step(self, batch, batch_idx):
        src, tgt = batch
        preds = self.forward(src)  # Run the model on the source data
        bleu_score = self.calculate_bleu(preds, tgt)  # Custom BLEU calculation function
        self.log("test_bleu", bleu_score, prog_bar=True)  # Log the BLEU score
        return {"test_bleu": bleu_score}

    def calculate_bleu(self, preds, targets):
        # Convert predictions and targets to text and calculate BLEU score
        return self.bleu_metric(preds, targets)  # Use the BLEUScore metric from torchmetrics

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.model.parameters(), lr=1e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.7)
        return [optimizer], [scheduler]


# PyTorch Lightning Trainer setup
wandb_logger = WandbLogger(project="eng2fre-translation")

# Early stopping setup (optional)
early_stopping = EarlyStopping(monitor='val_loss', patience=3)

# Model parameters (adjust these to fit your dataset)
vocab_size_src = 8000  # Replace with actual vocab size for English
vocab_size_tgt = 8000  # Replace with actual vocab size for French
emb_size = 512  # Embedding size, make sure it's divisible by num_heads
num_heads = 8   # Number of attention heads, should divide emb_size evenly
hidden_size = 2048
num_layers = 6  # Number of layers

# Instantiate the model
model = TranslationModel(vocab_size_src, vocab_size_tgt, emb_size, num_heads, num_layers)

# ------------------ TRAINING WITH PyTorch Lightning ------------------ #

trainer = pl.Trainer(
    logger=wandb_logger,  # Use W&B logger for experiment tracking
    max_epochs=10,        # Set the number of epochs
    accelerator="cpu",    # Specify the use of GPU (use "cpu" if not using GPU)
    precision=16,         # Use mixed precision if available
    callbacks=[early_stopping],  # Optional: early stopping
)

# Train the model
trainer.fit(model, train_dataloader, val_dataloader)

# ------------------ TESTING WITH PyTorch Lightning ------------------ #

trainer.test(model, test_dataloader)