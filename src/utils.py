import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
import numpy as np

import src.config as c


def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(" ")]


def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(c.EOS_token)
    return torch.tensor(indexes, dtype=torch.long).view(1, -1)


def tensorsFromPair(input_lang, output_lang, pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)


def get_dataloader(prepareData, batch_size):
    input_lang, output_lang, pairs = prepareData("eng", "fra", True)

    n = len(pairs)
    input_ids = np.zeros((n, c.MAX_LENGTH), dtype=np.int32)
    target_ids = np.zeros((n, c.MAX_LENGTH), dtype=np.int32)

    for idx, (inp, tgt) in enumerate(pairs):
        inp_ids = indexesFromSentence(input_lang, inp)
        tgt_ids = indexesFromSentence(output_lang, tgt)
        inp_ids.append(c.EOS_token)
        tgt_ids.append(c.EOS_token)
        input_ids[idx, : len(inp_ids)] = inp_ids
        target_ids[idx, : len(tgt_ids)] = tgt_ids

    train_data = TensorDataset(
        torch.LongTensor(input_ids), torch.LongTensor(target_ids)
    )

    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(
        train_data, sampler=train_sampler, batch_size=batch_size
    )
    return input_lang, output_lang, train_dataloader
