import copy
import torch
import torch.nn as nn
from model._abstract_model import SequentialRecModel
from model._modules import LayerNorm, FeedForward
import os
import numpy as np
import math

class FourierRecModel(SequentialRecModel):
    def __init__(self, args):
        super().__init__(args)
        print("Using FourierRec")
        self.args = args
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.item_encoder = FourierRecEncoder(args)
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


class FourierRecEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        block = FourierRecBlock(args)
        self.blocks = nn.ModuleList([copy.deepcopy(block) for _ in range(args.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=False, save=False):
        all_encoder_layers = [ hidden_states ]
        for layer_module in self.blocks:
            hidden_states = layer_module(hidden_states, attention_mask, save=save)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states) # hidden_states => torch.Size([256, 50, 64])
        return all_encoder_layers


class FourierRecBlock(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.layer = FourierRecRecLayer(args)
        self.feed_forward = FeedForward(args)

    def forward(self, hidden_states, attention_mask, save=False):
        layer_output = self.layer(hidden_states, attention_mask, save=save)
        feedforward_output = self.feed_forward(layer_output)
        return feedforward_output


class FourierRecRecLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.attention_layer = MultiHeadFourierAttention(args)

    def forward(self, input_tensor, attention_mask, save=False):

        actual_save = save and self.args.spectral
        hidden_states = self.attention_layer(input_tensor, attention_mask, save=actual_save)

        return hidden_states


class MultiHeadFourierAttention(nn.Module):
    def __init__(self, args):
        super().__init__()
        if args.hidden_size % args.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (args.hidden_size, args.num_attention_heads))

        self.args = args
        self.num_attention_heads = args.num_attention_heads
        self.c = args.c
        self.attention_head_size = int(args.hidden_size / args.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.sqrt_attention_head_size = math.sqrt(self.attention_head_size)

        self.query = nn.Linear(args.hidden_size, self.all_head_size)
        self.key = nn.Linear(args.hidden_size, self.all_head_size)
        self.value = nn.Linear(args.hidden_size, self.all_head_size)

        self.softmax = nn.Softmax(dim=-1)
        self.attn_dropout = nn.Dropout(args.attention_probs_dropout_prob)

        self.dense = nn.Linear(args.hidden_size, args.hidden_size)
        self.LayerNorm = nn.LayerNorm(args.hidden_size, eps=1e-12) # TODO
        self.out_dropout = nn.Dropout(args.hidden_dropout_prob)

        self.sqrt_beta = nn.Parameter(torch.randn(1, self.num_attention_heads, args.max_seq_length, 1))
        print("beta shape", self.sqrt_beta.shape)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x

    def forward(self, input_tensor, attention_mask, save=False):
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        query_layer = self.transpose_for_scores(mixed_query_layer).permute(0, 2, 1, 3)
        key_layer = self.transpose_for_scores(mixed_key_layer).permute(0, 2, 3, 1)
        value_layer = self.transpose_for_scores(mixed_value_layer).permute(0, 2, 1, 3)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer)

        # Fourier per head
        x = torch.fft.rfft(attention_scores, dim=-2, norm='ortho')
        low_pass = x
        low_pass[:, self.c:, :] = 0
        seq_len = attention_scores.shape[-1]
        low_pass = torch.fft.irfft(low_pass, n=seq_len, dim=-2, norm='ortho')
        high_pass = attention_scores - low_pass
        attention_scores = low_pass + (self.sqrt_beta**2) * high_pass

        attention_scores = attention_scores / self.sqrt_attention_head_size
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        # [batch_size heads seq_len seq_len] scores
        # [batch_size 1 1 seq_len]
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = self.softmax(attention_scores)


        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.attn_dropout(attention_probs)

        if save:
            iteration = 0
            file_path = os.path.join(self.args.save_path, self.args.data_name, f'{self.args.run_name}_spectral_{iteration}.npy')
            directory = os.path.join(self.args.save_path, self.args.data_name)
            if not os.path.exists(directory):
                os.makedirs(directory)
            while os.path.exists(file_path):
                iteration += 1
                file_path = os.path.join(self.args.save_path, self.args.data_name, f'{self.args.run_name}_spectral_{iteration}.npy')
            np.save(file_path, attention_probs.cpu().detach().numpy())

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states
