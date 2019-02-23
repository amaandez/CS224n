#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1h
import torch
import torch.nn as nn
import torch.nn.functional as F


class Highway(nn.Module):
    def __init__(self, e_word):
        super(Highway, self). __init__()
        self.highway_proj = nn.Linear(e_word, e_word, bias = True)
        self.highway_gate = nn.Linear(e_word, e_word, bias = True)

    def forward(self, x):
        x_proj = F.relu(self.highway_proj(x))
        x_gate = F.sigmoid(self.highway_gate(x))
        x_highway = torch.mul(x_proj, x_gate) + torch.mul(torch.add(torch.mul(x_gate, -1),1), x)
        return x_highway

### END YOUR CODE
