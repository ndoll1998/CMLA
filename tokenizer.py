import os
import re
from collections import OrderedDict

def tokenize(text, do_lower_case=True):
    # remove unnecessary whitespaces
    text = text.strip()
    # check if text is valid
    if (text is None) or (len(text) == 0):
        return []
    # do lower case
    if do_lower_case:
        text = text.lower()
    # whitespace tokenizate
    white_tokens = text.split()
    # separate punktuation from tokens
    tokens = []
    for token in white_tokens:
        # split on punctuation
        token_pieces = re.split(r"(\w+)", token)
        token_pieces = [piece for piece in token_pieces if len(piece) > 0]
        # add to tokens and mark pieces with ##
        tokens.extend(token_pieces[0:1] + ['##' + piece for piece in token_pieces[1:]])
    # return tokens
    return tokens

class WordTokenizer(object):

    def __init__(self, vocab, do_lower_case=True, unknown_token="[UNK]", padding_token="[PAD]"):
        # create empty vocabulary
        self.vocab_ = OrderedDict()
        self.do_lower_case = do_lower_case
        self.unknown_token = unknown_token
        self.padding_token = padding_token

        # if vocab is a path to a file
        if (type(vocab) is str) and os.path.isfile(vocab):
            # open file and read tokens
            with open(vocab, 'r', encoding='latin-1') as f:
                tokens = f.readlines()
            # initialize from tokens
            self.__init__(tokens)

        # if vocab is the vocabulary
        elif type(vocab) in (list, tuple):
            # add tokens to vocab
            self.add_tokens(vocab)

        # invalid argument
        else:
            raise RuntimeError("Vocab must be either a path to a vocab-file or the ordered vocabulary!")

        # make sure special tokens are in vocab
        if self.unknown_token not in self.vocab_:
            raise RuntimeError("Unkown token is not in vocab!")

    @property
    def pad_token_id(self):
        return self.convert_token_to_id(self.padding_token)
    @property
    def unk_token_id(self):
        return self.convert_token_to_id(self.unknown_token)
    @property
    def vocab(self):
        return list(self.vocab_.keys())

    def tokenize(self, text):
        return tokenize(text, do_lower_case=self.do_lower_case)

    def convert_token_to_id(self, token):
        # get token-id if contained in vocab else return unnknown token
        return self.vocab_.get(token.replace('##', ''), self.vocab_[self.unknown_token])

    def convert_tokens_to_ids(self, tokens):
        # convert all tokens to ids
        return [self.convert_token_to_id(token) for token in tokens]

    def add_token(self, token):
        token = token.rstrip("\n").replace('##', '')
        # check if token is already in vocab
        if token not in self.vocab_:
            # add to vocab
            self.vocab_[token] = len(self.vocab_)

    def add_tokens(self, tokens):
        # add all tokens to vocab
        for token in tokens:
            self.add_token(token)

    def __len__(self):
        return len(self.vocab_)

    def save(self, path):
        # save vocabulary to path
        with open(os.path.join(path, 'vocab.txt'), 'w+') as f:
            for token in self.vocab_:
                f.write(token + '\n')