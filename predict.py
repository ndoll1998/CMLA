import os
# import pytorch
import torch
# import model and tokenizer
from model import CMLA
from tokenizer import WordTokenizer

# model directory
model_dir = "results/SemEval2015"
# sample text
text = "The ambience is nice for conversation."
text = "The staff was really nice but the food was disgusting!"
# text = "In the summer months, the back garden area is really nice."
# text = "Das Essen war sehr lecker."

# load tokenizer and model
print("Loading Tokenizer and Model...")
tokenizer = WordTokenizer(os.path.join(model_dir, 'vocab.txt'))
model = CMLA.load(model_dir)
model.eval()
# tokenize text
tokens = tokenizer.tokenize(text)
token_ids = tokenizer.convert_tokens_to_ids(tokens)
# convert to tensor and pass through model
token_ids = torch.LongTensor([token_ids])
aspect_logits, opinion_logits = model.forward(token_ids)
# get predictions from logits
aspect_predicts = aspect_logits[0, :].max(dim=-1)[1]
opinion_predicts = opinion_logits[0, :].max(dim=-1)[1]

print(tokens)
print(aspect_predicts)
print(opinion_predicts)
