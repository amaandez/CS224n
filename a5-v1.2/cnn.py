#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1i
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, f, e_char, kernel_size = 5):
        super(CNN, self).__init__()
        self.conv_layer = nn.Conv1d(e_char ,f, kernel_size)
        #self.max_pool = nn.MaxPool2d(kernel_size)

    def forward(self, x_reshaped):
        x_conv_output = torch.max(F.relu(self.conv_layer(x_reshaped)), 2)[0]
        return x_conv_output


### END YOUR CODE
