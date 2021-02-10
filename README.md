# BERT GloVe Embeddings

## What is this?

In certain applications, often in the context of adversarial text attacks targeting BERT, it is desirable to have embeddings for BERT's word pieces that are in GloVe space. The reason for this is because BERT embeddings are not trained to keep semantically similar tokens close in the embedding space. In Section 4.2 of "Rewriting Meaningful Sentences via Conditional BERT Sampling and an application on fooling text classifiers", the authors describe a method to train these kinds of embeddings. This repo is an implementation of their method in PyTorch.

## Running the code

First you'll need the required packages in `requirements.txt`.

```
torch==1.7.1
torchtext==0.8.1
tqdm==4.45.0
transformers==4.3.2
```

Then run `python train.py --help` for all the available options.

## Pre-trained BERT GloVe Embeddings

`bert_glove_embeddings.pt` contains some pretrained embeddings for anyone who doesn't want to train their own. They were specifically trained using GloVe's 200 dimensional, 6 billion token embeddings.

`losses.json` contains the ![formula](https://render.githubusercontent.com/render/math?math=L_1) losses for the pretrained embeddings over each epoch to help you determine if they've been trained too little or too much for your purposes.
