"""
    Paper: https://arxiv.org/abs/2010.11869
    Section 4.2: Training Word Piece Embeddings in GloVe Space
    Description:
        BERT Embeddings are not trained to keep semantically similar tokens close in
        the embedding space. To reconcile this, we can train our own embeddings that
        correspond to each token in BERT's word-piece vocabulary that lie in GloVe space.
        This is useful for tasks such as finding similar tokens during adversarial attacks.
"""


import argparse
from itertools import chain
import json

import torch
from torch.optim import SGD

from torchtext.data import Field
from torchtext.datasets import IMDB
from torchtext.vocab import GloVe

from transformers import BertModel, BertTokenizer

from tqdm import tqdm, trange


def set_seed(seed, set_torch_cuda=False):
    torch.manual_seed(seed)

    if set_torch_cuda:
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train(
    batch_size,
    bert_model,
    epochs,
    glove_dataset,
    glove_dim,
    save_file,
    seed,
    stats_file,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(seed, device == "cuda")

    glove = GloVe(name=glove_dataset, dim=glove_dim)
    glove_embeddings = glove.vectors
    glove_ids_to_tokens = glove.itos
    glove_tokens_to_ids = glove.stoi

    bert_tokenizer = BertTokenizer.from_pretrained(bert_model)
    bert = BertModel.from_pretrained(bert_model)
    bert_embeddings = bert.embeddings.word_embeddings.weight

    # E (GloVe embeddings): (N, d)
    E = glove_embeddings.to(device)
    # T (Word piece tokenization indicator): (N, 30k)
    T = torch.zeros(
        (glove_embeddings.size(0), bert_embeddings.size(0)),
        dtype=torch.bool,
        device=device,
    )
    # E' (BERT GloVe Embeddings): (30k, d)
    E_prime = torch.rand(
        (bert_embeddings.size(0), glove_embeddings.size(1)),
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )

    """
        Initialize T such that T[i, j] is 1 if word i is tokenized (using GloVe) to the 
        j-th word piece (using BERT) and 0 otherwise.

        Example:
            'handyman' is tokenized by BERT to 'handy' and '##man'.
            'handyman' is token 29172 in GloVe while 'handy' and '##man' are
            tokens 18801 and 2386 respectively so T[29172, 18801] and T[29172, 2386]
            are 1 while the rest of T[29172, :] is 0
    """
    for glove_id, glove_token in enumerate(
        tqdm(glove_ids_to_tokens, desc="Building T")
    ):
        bert_tokens = bert_tokenizer.tokenize(glove_token)

        # If any of the word pieces aren't in GloVe's vocabulary, we'll ignore them
        # This really only applies to certain Unicode characters
        if "[UNK]" in bert_tokens:
            continue

        bert_ids = bert_tokenizer.convert_tokens_to_ids(bert_tokens)

        T[glove_id, bert_ids] = True

    # Load IMDB Corpus
    text = Field(
        lower=BertTokenizer.pretrained_init_configuration[bert_model]["do_lower_case"],
        tokenize="spacy",
    )
    label = Field(sequential=False)
    train, test = IMDB.splits(text, label)
    corpus = torch.tensor(
        [
            glove_tokens_to_ids[token]
            for batch in chain(train, test)
            for token in batch.text
            if token in glove_tokens_to_ids
        ],
        dtype=torch.long,
        device=device,
    )

    losses = []

    # Train
    optimizer = SGD([E_prime], lr=1e-4)
    for epoch in trange(epochs, desc="Training"):
        random_indices = torch.randperm(corpus.size(0))[:batch_size]
        random_samples = corpus[random_indices]

        optimizer.zero_grad()

        loss = torch.sum(
            torch.abs(
                E[random_samples] - T[random_samples].type(torch.float32) @ E_prime
            )
        )
        losses.append(loss.item())

        loss.backward()
        optimizer.step()

        if stats_file is not None:
            with open(stats_file, "w+") as f:
                json.dump(losses, f)

    torch.save(E_prime, save_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch-size",
        default=5_000,
        help="The number of words to sample per training batch (default: %(default)s).",
        type=int,
    )
    parser.add_argument(
        "--bert-model",
        choices=BertTokenizer.pretrained_init_configuration.keys(),
        default="bert-base-uncased",
        help="The name of the pre-trained BERT model to use for tokenization (see: https://huggingface.co/transformers/pretrained_models.html, default: %(default)s)",
    )
    parser.add_argument(
        "--epochs",
        default=10_000,
        help="The number of training epochs to perform (default: %(default)s)",
        type=int,
    )
    parser.add_argument(
        "--glove-dim",
        default=200,
        help="The width of the GloVe word embeddings (default: %(default)s).",
        type=int,
    )
    parser.add_argument(
        "--glove-dataset",
        choices=GloVe.url.keys(),
        default="6B",
        help="The dataset GloVe trained on (see: https://nlp.stanford.edu/projects/glove/, default: %(default)s).",
    )
    parser.add_argument(
        "--save-file",
        default="bert_glove_embeddings.pt",
        help="The filename to save the trained embedding tensor to (default: %(default)s).",
    )
    parser.add_argument(
        "--seed",
        default=0,
        help="The seed used to for all torch randomization methods (default: %(default)s).",
        type=int,
    )
    parser.add_argument(
        "--stats-file", help="Where to save the per-epoch losses while training."
    )
    args = parser.parse_args()
    train(**args.__dict__)
