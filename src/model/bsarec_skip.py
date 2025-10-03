import copy
import torch
import torch.nn as nn
from model._abstract_model import SequentialRecModel
from model._modules import LayerNorm, FeedForward, MultiHeadAttention
import os
import numpy as np
from torch import Tensor
from typing import List

class BSARecModelPadding(SequentialRecModel):
    def __init__(self, args):
        super(BSARecModelPadding, self).__init__(args)
        self.args = args
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.item_encoder = BSARecEncoderPadding(args)
        self.apply(self.init_weights)

    def forward(self, input_ids, user_ids=None, all_sequence_output=False, save=False):
        extended_attention_mask = self.get_attention_mask(input_ids)
        sequence_emb = self.add_position_embedding(input_ids)
        item_encoded_layers = self.item_encoder(sequence_emb,
                                                extended_attention_mask,
                                                output_all_encoded_layers=True,
                                                save=save
                                                )
        if all_sequence_output:
            sequence_output = item_encoded_layers
        else:
            sequence_output = item_encoded_layers[-1]

        return sequence_output

    def calculate_loss(self, input_ids, answers, neg_answers, same_target, user_ids):
        seq_output = self.forward(input_ids)
        seq_output = seq_output[:, -1, :]
        item_emb = self.item_embeddings.weight
        logits = torch.matmul(seq_output, item_emb.transpose(0, 1))
        loss = nn.CrossEntropyLoss()(logits, answers)

        return loss

class BSARecEncoderPadding(nn.Module):
    def __init__(self, args):
        super(BSARecEncoderPadding, self).__init__()
        self.args = args
        block = BSARecBlockPadding(args)
        # self.blocks = nn.ModuleList([copy.deepcopy(block) for _ in range(args.num_hidden_layers)])
        # first layer has first_layer=True. Other do not
        self.blocks = nn.ModuleList([copy.deepcopy(block) for _ in range(args.num_hidden_layers)])
        self.blocks[0] = BSARecBlockPadding(args, first_layer=True)

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=False, save=False):
        all_encoder_layers = [ hidden_states ]
        for layer_module in self.blocks:
            hidden_states = layer_module(hidden_states, attention_mask, save=save)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states) # hidden_states => torch.Size([256, 50, 64])
        return all_encoder_layers

class BSARecBlockPadding(nn.Module):
    def __init__(self, args, first_layer=False):
        super(BSARecBlockPadding, self).__init__()
        self.layer = BSARecLayerPadding(args, first_layer=first_layer)
        self.feed_forward = FeedForward(args)

    def forward(self, hidden_states, attention_mask, save=False):
        layer_output = self.layer(hidden_states, attention_mask, save=save)
        feedforward_output = self.feed_forward(layer_output)
        return feedforward_output

class BSARecLayerPadding(nn.Module):
    def __init__(self, args, first_layer=False):
        super(BSARecLayerPadding, self).__init__()
        self.args = args
        self.attention_layer = MultiHeadAttention(args)
        self.alpha = args.alpha

    def forward(self, input_tensor, attention_mask, save=False):

        actual_save = save and self.args.spectral
        gsp = self.attention_layer(input_tensor, attention_mask, save=actual_save)
        hidden_states = gsp + input_tensor



        return hidden_states
    
