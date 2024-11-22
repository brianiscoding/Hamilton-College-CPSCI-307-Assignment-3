import torch
from pytorch_lightning import LightningModule
from a3_train import Seq2SeqModel, TranslationDataset
import sentencepiece as spm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_tokenizers(en_bpe_path, fr_bpe_path):
    en_tokenizer = spm.SentencePieceProcessor()
    en_tokenizer.load(en_bpe_path)
    fr_tokenizer = spm.SentencePieceProcessor()
    fr_tokenizer.load(fr_bpe_path)
    return en_tokenizer, fr_tokenizer

def load_model(ckpt_path):
    model = Seq2SeqModel.load_from_checkpoint("model-epoch=09-val_loss=1.03.ckpt")
    model.to(device)
    model.eval()
    return model

def translate_with_beam_search(model, en_tokenizer, fr_tokenizer, input_text, max_length=50, beam_size=3):
    # Step 1: Tokenize English input and move to device
    input_ids = en_tokenizer.encode(input_text, out_type=int)
    input_tensor = torch.tensor([input_ids]).to(device)

    # Step 2: Initialize beams with <sos> token
    beams = [(torch.tensor([fr_tokenizer.bos_id()], device=device), 0.0)]  # (sequence, score)

    # Step 3: Perform beam search
    for _ in range(max_length):
        new_beams = []
        for seq, score in beams:
            with torch.no_grad():
                output = model(input_tensor, seq.unsqueeze(0))
                logits = output[0, -1]  # Get logits for the last token
                probs = torch.softmax(logits, dim=-1)  # Convert logits to probabilities

            # Get top-k tokens and their probabilities
            topk_probs, topk_tokens = probs.topk(beam_size)

            for prob, token in zip(topk_probs, topk_tokens):
                new_seq = torch.cat([seq, token.unsqueeze(0)])  # Append token to sequence
                new_score = score + torch.log(prob).item()  # Accumulate log-probability
                new_beams.append((new_seq, new_score))

        # Keep top-k beams
        beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_size]

        # Stop if all beams end with <eos>
        if all(seq[-1].item() == fr_tokenizer.eos_id() for seq, _ in beams):
            break

    # Step 4: Decode the best sequence
    best_seq = beams[0][0]
    translated_text = fr_tokenizer.decode(best_seq[1:].tolist())  # Exclude <sos> token
    return translated_text


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Translate text using a trained model.")
    parser.add_argument("ckpt_path", type=str, help="Path to the checkpoint file.")
    parser.add_argument("en_bpe_path", type=str, help="Path to the English BPE model file.")
    parser.add_argument("fr_bpe_path", type=str, help="Path to the French BPE model file.")
    parser.add_argument("--input", type=str, required=True, help="Input text to translate.")

    args = parser.parse_args()

    en_tokenizer, fr_tokenizer = load_tokenizers(args.en_bpe_path, args.fr_bpe_path)
    model = load_model(args.ckpt_path)
    translated_text = translate_with_beam_search(model, en_tokenizer, fr_tokenizer, args.input)
    print("Translated Text:", translated_text)



