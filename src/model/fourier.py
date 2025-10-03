import copy
import torch
import torch.nn as nn
from model._abstract_model import SequentialRecModel
from model._modules import LayerNorm, FeedForward


class FourierModel(SequentialRecModel):
    def __init__(self, args):
        super(FourierModel, self).__init__(args)
        self.args = args
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)

        # pass max_seq_len and gamma down to encoder
        self.item_encoder = FourierEncoder(args)

        self.apply(self.init_weights)

    def forward(self, input_ids, user_ids=None, all_sequence_output=False, save=False):
        extended_attention_mask = self.get_attention_mask(input_ids)
        sequence_emb = self.add_position_embedding(input_ids)
        item_encoded_layers = self.item_encoder(
            sequence_emb,
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
        seq_output = seq_output[:, -1, :]                          # [B, d]
        item_emb = self.item_embeddings.weight                     # [V, d]
        logits = torch.matmul(seq_output, item_emb.t())            # [B, V]
        loss = nn.CrossEntropyLoss()(logits, answers)
        return loss


class FourierEncoder(nn.Module):
    def __init__(self, args):
        super(FourierEncoder, self).__init__()
        self.args = args
        block = FourierBlock(args)
        self.blocks = nn.ModuleList(
            [copy.deepcopy(block) for _ in range(args.num_hidden_layers)]
        )

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=False, save=False):
        all_layers = [hidden_states]
        for blk in self.blocks:
            hidden_states = blk(hidden_states, attention_mask, save=save)
            if output_all_encoded_layers:
                all_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_layers.append(hidden_states)
        return all_layers


class FourierBlock(nn.Module):
    def __init__(self, args):
        super(FourierBlock, self).__init__()
        self.layer = FourierLayer(args)
        self.feed_forward = FeedForward(args)

    def forward(self, hidden_states, attention_mask, save=False):
        x = self.layer(hidden_states, attention_mask, save=save)
        x = self.feed_forward(x)
        return x


class FourierLayer(nn.Module):
    def __init__(self, args):
        super(FourierLayer, self).__init__()
        self.args = args
        self.filter_attention = FilterHeadAttention(args)
        self.alpha = args.alpha

    def forward(self, x, attention_mask, save=False):
        # only GSP branch for now
        out = self.filter_attention(x, save=save)
        return out


class FilterHeadAttention(nn.Module):
    def __init__(self, args):
        super(FilterHeadAttention, self).__init__()
        self.filter_layer = FrequencyLayer_Fourier(args)

        self.dense = nn.Linear(args.hidden_size, args.hidden_size)
        self.LayerNorm1 = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout1 = nn.Dropout(args.hidden_dropout_prob)

        self.LayerNorm2 = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout2 = nn.Dropout(args.hidden_dropout_prob)

    def forward(self, x, save=False):
        # time-frequency filtering
        y = self.filter_layer(x, save=save)             # [B, T, d]
        y = self.dropout2(y)
        y = self.LayerNorm2(x + y)

        # feed into a “self-attention” style dense & residual
        y2 = self.dense(y)
        y2 = self.dropout1(y2)
        y2 = self.LayerNorm1(y2 + y)
        return y2


class FrequencyLayer_Fourier(nn.Module):
    def __init__(self, args):
        super(FrequencyLayer_Fourier, self).__init__()
        self.out_dropout = nn.Dropout(args.hidden_dropout_prob)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.c = args.c // 2 + 1
        self.sqrt_beta = nn.Parameter(torch.randn(1, 1, args.hidden_size))
        self.args = args

    def forward(self, input_tensor, save=False):
        # [batch, seq_len, hidden]
        batch, seq_len, hidden = input_tensor.shape
        x = torch.fft.rfft(input_tensor, dim=1, norm='ortho')
        
        low_pass = x[:]
        low_pass[:, self.c:, :] = 0
        low_pass = torch.fft.irfft(low_pass, n=seq_len, dim=1, norm='ortho')
        high_pass = input_tensor - low_pass
        sequence_emb_fft = low_pass + (self.sqrt_beta**2) * high_pass



        hidden_states = self.out_dropout(sequence_emb_fft)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states
