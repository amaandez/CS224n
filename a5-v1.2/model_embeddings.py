#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
model_embeddings.py: Embeddings for the NMT model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
Anand Dhoot <anandd@stanford.edu>
Michael Hahn <mhahn2@stanford.edu>
"""

import torch.nn as nn
import torch

# Do not change these imports; your module names should be
#   `CNN` in the file `cnn.py`
#   `Highway` in the file `highway.py`
# Uncomment the following two imports once you're ready to run part 1(j)

from cnn import CNN
from highway import Highway

# End "do not change"

class ModelEmbeddings(nn.Module):
    """
    Class that converts input words to their CNN-based embeddings.
    """
    def __init__(self, embed_size, vocab):
        """
        Init the Embedding layer for one language
        @param embed_size (int): Embedding size (dimensionality) for the output
        @param vocab (VocabEntry): VocabEntry object. See vocab.py for documentation.
        """
        super(ModelEmbeddings, self).__init__()

        ## A4 code
        #pad_token_idx = vocab.src['<pad>']
        #self.embeddings = nn.Embedding(len(vocab.src), embed_size, padding_idx=pad_token_idx)
        ## End A4 code

        ### YOUR CODE HERE for part 1j

        self.embed_size = embed_size
        pad_token_idx = vocab.char2id['<pad>']

        self.embeddings = nn.Embedding(num_embeddings=len(vocab.char2id), embedding_dim = 50, padding_idx=pad_token_idx)
        self.highway = Highway(e_word = embed_size)

        self.cnn = CNN(f= embed_size, e_char = 50)
        self.dropout = nn.Dropout(.3)


        ### END YOUR CODE

    def forward(self, input):
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.
        @param input: Tensor of integers of shape (sentence_length, batch_size, max_word_length) where
            each integer is an index into the character vocabulary

        @param output: Tensor of shape (sentence_length, batch_size, embed_size), containing the
            CNN-based embeddings for each word of the sentences in the batch
        """
        ## A4 code
        #output = self.embeddings(input)
        #return output
        ## End A4 code

        ### YOUR CODE HERE for part 1j
        #embeddings = self.embeddings(input)
        #print (embeddings.shape)
        #torch.Size([10, 5, 21, 3])
        #shape = embeddings.shape
        #x_reshaped = torch.reshape(embeddings, (shape[0] * shape[1], shape[2], shape[3]))
        #embeddings = embeddings.permute(0,2,1)

        #x_reshaped = torch.reshape(embeddings, (shape[0] * shape[1], shape[2], shape[3]))
        #x_conv_out = self.myCNN.forward(x_reshaped)

        #x_highway = self.myHighway(x_conv_out)
        #expand it back out

        #x_word_embed = self.dropout(x_highway)
        input_shape = input.shape
        input_reshaped = torch.reshape(input, (input_shape[0] * input_shape[1], input_shape[2]))
        x_padded = self.embeddings(input_reshaped)
        x_reshaped = x_padded.permute(0,2,1)

        x_conv_out = self.cnn.forward(x_reshaped)
        x_highway = self.highway(x_conv_out)

        x_highway_reshaped = torch.reshape(x_highway, (input_shape[0], input_shape[1], x_highway.shape[1]))
        x_word_embed = self.dropout(x_highway_reshaped)
        return x_word_embed

        ### END YOUR CODE
