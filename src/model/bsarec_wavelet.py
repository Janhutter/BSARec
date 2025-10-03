import copy
import torch
import torch.nn as nn
from model._abstract_model import SequentialRecModel
from model._modules import LayerNorm, FeedForward, MultiHeadAttention
import os
import numpy as np
# from pytorch_wavelets import DWT1DForward, DWT1DInverse  # 1D Wavelet Transform


class BSARec_WaveletModel(SequentialRecModel):
    def __init__(self, args):
        super(BSARec_WaveletModel, self).__init__(args)
        self.args = args
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.item_encoder = BSARec_WaveletEncoder(args)
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

class BSARec_WaveletEncoder(nn.Module):
    def __init__(self, args):
        super(BSARec_WaveletEncoder, self).__init__()
        self.args = args
        block = BSARec_WaveletBlock(args)
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

class BSARec_WaveletBlock(nn.Module):
    def __init__(self, args):
        super(BSARec_WaveletBlock, self).__init__()
        self.layer = BSARec_WaveletLayer(args)
        self.feed_forward = FeedForward(args)

    def forward(self, hidden_states, attention_mask, save=False):
        layer_output = self.layer(hidden_states, attention_mask, save=save)
        feedforward_output = self.feed_forward(layer_output)
        return feedforward_output

class BSARec_WaveletLayer(nn.Module):
    def __init__(self, args):
        super(BSARec_WaveletLayer, self).__init__()
        self.args = args
        self.filter_layer = FrequencyLayer_Wavelet(args)
        self.attention_layer = MultiHeadAttention(args)
        self.alpha = args.alpha

    def forward(self, input_tensor, attention_mask, save=False):

        actual_save = save and self.args.spectral
        dsp = self.filter_layer(input_tensor, save=actual_save)
        gsp = self.attention_layer(input_tensor, attention_mask, save=actual_save)
        hidden_states = self.alpha * dsp + ( 1 - self.alpha ) * gsp

        return hidden_states
    
class FrequencyLayer_Wavelet(nn.Module):
    def __init__(self, args):
        super(FrequencyLayer_Wavelet, self).__init__()
        self.out_dropout = nn.Dropout(args.hidden_dropout_prob)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.c = args.c // 2 + 1
        self.sqrt_beta = nn.Parameter(torch.randn(1, 1, args.hidden_size))
        self.args = args
        self.wave = getattr(args, 'wavelet', args.wavelet)
        self.mode = getattr(args, 'wavelet_mode', 'zero')  # or 'symmetric', etc.

        self.dwt = DWT1DForward(J=1, wave=self.wave, mode=self.mode)   # Decomposition
        self.idwt = DWT1DInverse(wave=self.wave, mode=self.mode)       # Reconstruction

    def forward(self, input_tensor, save=False):
        # input_tensor: [batch, seq_len, hidden]
        B, T, H = input_tensor.shape

        # Transpose for DWT: [B, H, T]
        x = input_tensor.permute(0, 2, 1)

        # Apply DWT: returns (approx, [detail])
        approx, detail = self.dwt(x)

        # Reconstruct low-pass (approx only)
        zeros_detail = [torch.zeros_like(d) for d in detail]
        low_pass = self.idwt((approx, zeros_detail))

        # Reconstruct high-pass (detail only)
        zero_approx = torch.zeros_like(approx)
        high_pass = self.idwt((zero_approx, detail))

        # Transpose back to [B, T, H]
        low_pass = low_pass.permute(0, 2, 1)
        high_pass = high_pass.permute(0, 2, 1)

        sequence_emb_wavelet = low_pass + (self.sqrt_beta ** 2) * high_pass

        if save:
            iteration = 0
            directory = os.path.join(self.args.save_path, self.args.data_name)
            os.makedirs(directory, exist_ok=True)
            file_path = os.path.join(directory, f'{self.args.run_name}_filter_{iteration}.npy')
            while os.path.exists(file_path):
                iteration += 1
                file_path = os.path.join(directory, f'{self.args.run_name}_filter_{iteration}.npy')
            np.save(file_path, sequence_emb_wavelet.cpu().detach().numpy())

        hidden_states = self.out_dropout(sequence_emb_wavelet)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states
