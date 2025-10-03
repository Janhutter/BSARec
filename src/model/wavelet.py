import copy
import torch
import torch.nn as nn
from model._abstract_model import SequentialRecModel
from model._modules import LayerNorm, FeedForward
# from pytorch_wavelets import DWT1DForward, DWT1DInverse  # 1D Wavelet Transform


class WaveletModel(SequentialRecModel):
    def __init__(self, args):
        super(WaveletModel, self).__init__(args)
        self.args = args
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)

        # pass max_seq_len and gamma down to encoder
        self.item_encoder = WaveletEncoder(args)

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


class WaveletEncoder(nn.Module):
    def __init__(self, args):
        super(WaveletEncoder, self).__init__()
        self.args = args
        block = WaveletBlock(args)
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


class WaveletBlock(nn.Module):
    def __init__(self, args):
        super(WaveletBlock, self).__init__()
        self.layer = WaveletLayer(args)
        self.feed_forward = FeedForward(args)

    def forward(self, hidden_states, attention_mask, save=False):
        x = self.layer(hidden_states, attention_mask, save=save)
        x = self.feed_forward(x)
        return x


class WaveletLayer(nn.Module):
    def __init__(self, args):
        super(WaveletLayer, self).__init__()
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
        self.filter_layer = FilterLayer_Wavelet(args)

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


class FilterLayer_Wavelet(nn.Module):
    def __init__(self, args):
        super(FilterLayer_Wavelet, self).__init__()
        self.args = args
        self.gamma = 3
        self.hidden_size = args.hidden_size
        self.max_seq = args.max_seq_length

        self.dwt = DWT1DForward(J=self.gamma, wave=args.wavelet, mode=args.wavelet_mode)
        self.idwt = DWT1DInverse(wave=args.wavelet, mode=args.wavelet_mode)

        with torch.no_grad():
            dummy = torch.zeros(1, self.hidden_size, self.max_seq)
            approx, details = self.dwt(dummy)
            self.level_lens = [d_i.shape[-1] for d_i in details]

        self.W = nn.ParameterList([
            nn.Parameter(torch.randn(l_i, self.hidden_size) * 0.02)
            for l_i in self.level_lens
        ])
        self.r = nn.Parameter(torch.ones(self.hidden_size))

        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.LayerNorm = LayerNorm(self.hidden_size, eps=1e-12)

    def forward(self, x, save=False):
            B, T, D = x.shape
            x_ch = x.permute(0, 2, 1)

            A_gamma, details = self.dwt(x_ch)
            new_details = []
            for i, D_i in enumerate(details):
                tmp = D_i.permute(0, 2, 1)

                tmp = tmp * self.W[i].unsqueeze(0)

                tmp = tmp.permute(0, 2, 1)

                tmp = tmp * (self.r.unsqueeze(0).unsqueeze(-1) ** 2)

                new_details.append(tmp)


            x_rec = self.idwt((A_gamma, new_details))
            x_rec = x_rec[..., :T]

            hidden = x_rec.permute(0, 2, 1)

            hidden = self.dropout(hidden)
            hidden = self.LayerNorm(hidden + x)

            return hidden
