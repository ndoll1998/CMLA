import os
import json
# import numpy
import numpy as np
# import pytorch
import torch
# import model and tokenizer
from model import CMLA
from tokenizer import WordTokenizer, tokenize
# import f1-score
from sklearn.metrics import f1_score
# other imports
from tqdm import tqdm
from time import time

# files
train_data_file = "data/train.json"
test_data_file = "data/test.json"
# general
epochs = 1
batch_size = 5
max_seq_length = 64
# optimizer
learning_rate = 1e-4
weight_decay = 0
# tokenizer
do_lower_case = True

# prepare data
print("Preparating data...")

def load_data(fpath):
    vocab = set()
    all_tokenized_text, all_labels_a, all_labels_o = [], [], []

    def mark_bio(sequence, subsequences):
        # mark all subsequences in sequence in bio-scheme
        marked = [0] * len(sequence)
        for subseq in subsequences:
            for i in range(len(sequence) - len(subseq)):
                if sequence[i:i+len(subseq)] == subseq:
                    # mark b = 1 and i = 2
                    marked[i] = 1
                    marked[i+1:i+len(subseq)] = [2] * (len(subseq)-1)
        return marked

    # load data
    with open(fpath, 'r', encoding='utf-8') as f:
        data = json.loads(f.read())
    # preprocess data
    for text, aspects, opinions in tqdm(data):
        # tokenize text and match sequence length
        tokenized_text = tokenize(text, do_lower_case)[:max_seq_length]
        tokenized_text += ['[PAD]'] * (max_seq_length - len(tokenized_text))
        # tokenize aspects and opinions
        tokenized_aspects = map(tokenize, aspects)
        tokenized_opinions = map(tokenize, opinions)
        # create labels
        labels_a = mark_bio(tokenized_text, tokenized_aspects)
        labels_o = mark_bio(tokenized_text, tokenized_opinions)
        # update vocab
        vocab = vocab.union(tokenized_text)
        # add to lists
        all_tokenized_text.append(tokenized_text)
        all_labels_a.append(labels_a)
        all_labels_o.append(labels_o)
    # return
    return vocab, all_tokenized_text, all_labels_a, all_labels_o
# load data and unpdate vocab
vocab, train_text_tokens, train_labels_a, train_labels_o = load_data(train_data_file)
_, test_text_tokens, test_labels_a, test_labels_o = load_data(test_data_file)
vocab = list(vocab.union(['[UNK]', '[PAD]']))

# create tokenizer and model
print("Create Tokenizer and Model...")
tokenizer = WordTokenizer(vocab, do_lower_case=do_lower_case)
model = CMLA(50, 3, len(tokenizer), de=200, cs=3, K=20, l=2, pad_id=tokenizer.pad_token_id)
# optimizer and criterium
optim = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
criterium = torch.nn.CrossEntropyLoss()

# create input ids
print("Create Dataloader...")

def create_dataset(all_tokens, all_labels_a, all_labels_o):
    # convert tokens to ids
    all_ids = list(map(tokenizer.convert_tokens_to_ids, all_tokens))
    # create tensors
    all_ids = torch.LongTensor(all_ids)
    all_labels_a = torch.LongTensor(all_labels_a)
    all_labels_o = torch.LongTensor(all_labels_o)
    # create dataset
    return torch.utils.data.TensorDataset(all_ids, all_labels_a, all_labels_o)
# create dataloaders
train_dataloader = torch.utils.data.DataLoader(
    create_dataset(train_text_tokens, train_labels_a, train_labels_o), batch_size=batch_size, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(
    create_dataset(test_text_tokens, test_labels_a, test_labels_o), batch_size=batch_size, shuffle=False)

# train
print("Training Model...")

start = time()
for e in range(epochs):
    # train model
    model.train()
    running_loss = 0

    for step, (input_ids, targets_a, targets_o) in enumerate(train_dataloader, 1):
        # TODO: randomly add unknown-tokens where no paddings are
        # pass through model
        logits_a, logits_o = model.forward(input_ids)
        # compute loss
        loss_a = criterium(logits_a.view(-1, 3), targets_a.view(-1))
        loss_o = criterium(logits_o.view(-1, 3), targets_o.view(-1))
        loss = loss_a + loss_o
        # update model parameters
        optim.zero_grad()
        loss.backward()
        optim.step()
        # update running loss and log
        running_loss += loss.item()
        print("Epoch {0}\t - Step {1}/{2}\t - Loss {3:.04f}\t - Time {4:.04f}s".format(
            e, step, len(train_dataloader), running_loss/step, time() - start), end='\r')

    # compute average training loss
    train_loss = running_loss / len(train_dataloader)

    # evaluate model
    model.eval()
    running_loss = 0

    with torch.no_grad():

        # store predictions
        predicts_a, predicts_o = [], []

        for input_ids, t_a, t_o in test_dataloader:
            # pass through model
            logits_a, logits_o = model.forward(input_ids)
            # compute loss
            loss_a = criterium(logits_a.view(-1, 3), targets_a.view(-1))
            loss_o = criterium(logits_o.view(-1, 3), targets_o.view(-1))
            loss = loss_a + loss_o
            # update running loss
            running_loss += loss.item()
            # get predictions
            predicts_a += logits_a.view(-1, 3).max(dim=-1)[1].tolist()
            predicts_o += logits_o.view(-1, 3).max(dim=-1)[1].tolist()

    # compute average test loss
    test_loss = running_loss / len(test_dataloader)
    # compute f1-scores
    micro_f1_aspects = f1_score(predicts_a, sum(test_labels_a, []), average='micro')
    micro_f1_opinions = f1_score(predicts_o, sum(test_labels_o, []), average='micro')

    print("\nEpoch {0}\t - Train-Loss {1:.04f}\t - Test-Loss {2:.04f}\t - Aspect-F1 {3:.04f}\t - Opinion-F1 {4:.04f} - Time {5:.04f}".format(
        e, train_loss, test_loss, micro_f1_aspects, micro_f1_opinions, time() - start))