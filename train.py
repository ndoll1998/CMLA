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
from matplotlib import pyplot as plt

device = 'cuda:0'
# files
train_data_file = "data/SemEval2014/train.json"
test_data_file = "data/SemEval2014/test.json"
# general
epochs = 5
batch_size = 10
max_seq_length = 64
unknown_prob = 0.01
# optimizer
learning_rate = 1e-3
weight_decay = 0
# tokenizer
do_lower_case = True
gensim_embeddings_file = "C:/Users/Nicla/Documents/Datasets/WordEmbeddings/englishYelp.bin" # GoogleNews-vectors-negative300.bin"
embedding_dim = 200
# save directory
save_dir = "results/test"
os.makedirs(save_dir, exist_ok=True)

# prepare data
print("Preparing data...")

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
model = CMLA(50, 3, len(tokenizer), embedding_dim, 3, 20, 2, pad_id=tokenizer.pad_token_id)
# load pretrained embeddings
if gensim_embeddings_file is not None:
    print("Loading Pretrained Embeddings...")
    vocab = tokenizer.vocab
    if "german_deepset.bin" in gensim_embeddings_file:
        vocab = [("b'" + t + "'") for t in vocab]   # match tokens in german_deepset embeddings
    n_loaded = model.load_gensim_embeddings(gensim_embeddings_file, vocab, limit=100_000, binary=True)
    print("Loaded %i/%i vectors from pretrained embedding." % (n_loaded, len(tokenizer)))
# move model to device
model.to(device)

# optimizer and criterium
optim = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
criterium = torch.nn.CrossEntropyLoss()

# create input ids
print("Create Dataloaders...")

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

def place_unk(input_ids):
    # create mask
    rand_mask = np.random.uniform(0, 1, size=input_ids.size()) < unknown_prob
    padd_mask = (input_ids != tokenizer.pad_token_id).numpy()
    mask = torch.BoolTensor(rand_mask & padd_mask)
    # set values at mask to unknown tokens
    input_ids[mask] = tokenizer.unk_token_id
    # return manipulated ids
    return input_ids

start = time()
train_losses, test_losses = [], []
aspect_f1, opinion_f1 = [], []
# start training
for e in range(1, 1+epochs):
    # train model
    model.train()
    running_loss = 0

    for step, (input_ids, targets_a, targets_o) in enumerate(train_dataloader, 1):
        # randomly place unknown-tokens
        input_ids = place_unk(input_ids)
        # create padding mask
        mask = input_ids != tokenizer.pad_token_id
        targets_a = targets_a[mask].view(-1)
        targets_o = targets_o[mask].view(-1)
        # pass through model and apply mask
        logits_a, logits_o = model.forward(input_ids.to(device))
        logits_a, logits_o = logits_a[mask, :].view(-1, 3), logits_o[mask, :].view(-1, 3)
        # compute loss
        loss_a = criterium(logits_a, targets_a.to(device))
        loss_o = criterium(logits_o, targets_o.to(device))
        loss = (loss_a + loss_o)
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
    train_losses.append(train_loss)

    # evaluate model
    model.eval()
    running_loss = 0

    with torch.no_grad():

        # store predictions
        predicts_a, predicts_o = [], []
        targets_a, targets_o = [], []

        for input_ids, t_a, t_o in test_dataloader:
            # get padding mask and apply to targets
            mask = input_ids != tokenizer.pad_token_id
            t_a, t_o = t_a[mask].view(-1), t_o[mask].view(-1)
            # pass through model
            logits_a, logits_o = model.forward(input_ids.to(device))
            logits_a, logits_o = logits_a[mask, :].view(-1, 3), logits_o[mask, :].view(-1, 3)
            # compute loss
            loss_a = criterium(logits_a, t_a.to(device))
            loss_o = criterium(logits_o, t_o.to(device))
            loss = loss_a + loss_o
            # update running loss
            running_loss += loss.item()
            # get predictions
            predicts_a += logits_a.max(dim=-1)[1].tolist()
            predicts_o += logits_o.max(dim=-1)[1].tolist()
            # store targets
            targets_a += t_a.tolist()
            targets_o += t_o.tolist()

        # compute average test loss
        test_loss = running_loss / len(test_dataloader)
        test_losses.append(test_loss)
        # compute f1-scores
        macro_f1_aspects = f1_score(predicts_a, targets_a, average='macro')
        macro_f1_opinions = f1_score(predicts_o, targets_o, average='macro')

    # add to list
    aspect_f1.append(macro_f1_aspects)
    opinion_f1.append(macro_f1_opinions)

    print()
    print('Aspect-' + "['o', 'b', 'i']" + "  = " + str(f1_score(predicts_a, targets_a, average=None)))
    print('Opinion-' + "['o', 'b', 'i']" + " = " + str(f1_score(predicts_o, targets_o, average=None)))
    print("Epoch {0}\t - Train-Loss {1:.04f}\t - Test-Loss {2:.04f}\t - Aspect-F1 {3:.04f}\t - Opinion-F1 {4:.04f} - Time {5:.04f}\n".format(
        e, train_loss, test_loss, macro_f1_aspects, macro_f1_opinions, time() - start))

# plot losses and save figure
fig, ax = plt.subplots(1, 1)
ax.plot(train_losses)
ax.plot(test_losses)
ax.legend(["Train", "Test"])
ax.set(xlabel="Epoch", ylabel="Loss")
fig.savefig(os.path.join(save_dir, 'losses.png'), format='png')
# plot f1-scores and save figure
fig, ax = plt.subplots(1, 1)
ax.plot(aspect_f1)
ax.plot(opinion_f1)
ax.legend(["Aspect", "Opinion"])
ax.set(xlabel="Epoch", ylabel="F1-Score")
fig.savefig(os.path.join(save_dir, 'f1_score.png'), format='png')
# save model and vocab
model.save(save_dir)
tokenizer.save(save_dir)