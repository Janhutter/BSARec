import tqdm
import torch
import numpy as np

from torch.optim import Adam
from metrics import recall_at_k, ndcg_k
import wandb
from model.bsarec import FrequencyLayer
from model.bsarec_wavelet import FrequencyLayer_Wavelet
from model.fourierrec import MultiHeadFourierAttention
from utils import write_output
import os

class Trainer:
    def __init__(self, model, train_dataloader, eval_dataloader, test_dataloader, args, logger, wandb=True):
        super(Trainer, self).__init__()

        self.args = args
        self.logger = logger
        self.cuda_condition = torch.cuda.is_available() and not self.args.no_cuda
        self.device = torch.device("cuda" if self.cuda_condition else "cpu")

        self.model = model
        if self.cuda_condition:
            self.model.cuda()

        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.test_dataloader = test_dataloader

        betas = (self.args.adam_beta1, self.args.adam_beta2)
        self.optim = Adam(self.model.parameters(), lr=self.args.lr, betas=betas, weight_decay=self.args.weight_decay)

        self.logger.info(f"Total Parameters: {sum([p.nelement() for p in self.model.parameters()])}")
        self.wandb = wandb

    def train(self, epoch):
        self.iteration(epoch, self.train_dataloader, train=True)

    def valid(self, epoch):
        self.args.train_matrix = self.args.valid_rating_matrix
        return self.iteration(epoch, self.eval_dataloader, train=False)

    def test(self, epoch, save=False, spectral=False):
        self.args.train_matrix = self.args.test_rating_matrix

        scores, scores_string = self.iteration(epoch, self.test_dataloader, train=False, save=save, spectral=spectral)
        return scores, scores_string

    def save(self, file_name):
        torch.save(self.model.cpu().state_dict(), file_name)
        self.model.to(self.device)

    def load(self, file_name, map_location=None):
        original_state_dict = self.model.state_dict()
        self.logger.info(original_state_dict.keys())
        new_dict = torch.load(file_name, map_location=map_location)
        self.logger.info(new_dict.keys())
        for key in new_dict:
            if 'beta' in key:
                original_state_dict[key] = new_dict[key]
            else:
                original_state_dict[key] = new_dict[key]
        self.model.load_state_dict(original_state_dict)

    def predict_full(self, seq_out):
        # [item_num × hidden_size]
        test_item_emb = self.model.item_embeddings.weight
        # [batch × hidden_size]
        rating_pred = torch.matmul(seq_out, test_item_emb.transpose(0, 1))
        return rating_pred

    def get_full_sort_score(self, epoch, answers, pred_list):
        recall, ndcg = [], []
        for k in [5, 10, 15, 20]:
            recall.append(recall_at_k(answers, pred_list, k))
            ndcg.append(ndcg_k(answers, pred_list, k))
        post_fix = {
            "Epoch": epoch,
            "HR@5": '{:.4f}'.format(recall[0]), "NDCG@5": '{:.4f}'.format(ndcg[0]),
            "HR@10": '{:.4f}'.format(recall[1]), "NDCG@10": '{:.4f}'.format(ndcg[1]),
            "HR@20": '{:.4f}'.format(recall[3]), "NDCG@20": '{:.4f}'.format(ndcg[3])
        }
        self.logger.info(post_fix)

        return [recall[0], ndcg[0], recall[1], ndcg[1], recall[3], ndcg[3]], str(post_fix)

    def iteration(self, epoch, dataloader, train=True, save=False, spectral=False):
        str_code = "train" if train else "test"
        rec_data_iter = tqdm.tqdm(
            enumerate(dataloader),
            desc="Mode_%s:%d" % (str_code, epoch),
            total=len(dataloader),
            bar_format="{l_bar}{r_bar}"
        )

        if train:
            self.model.train()
            rec_loss = 0.0

            for i, batch in rec_data_iter:
                batch = tuple(t.to(self.device) for t in batch)
                user_ids, input_ids, answers, neg_answer, same_target = batch

                loss = self.model.calculate_loss(input_ids, answers, neg_answer, same_target, user_ids)
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                rec_loss += loss.item()

            post_fix = {
                "epoch": epoch,
                "rec_loss": '{:.4f}'.format(rec_loss / len(rec_data_iter)),
            }

            if (epoch + 1) % self.args.log_freq == 0:
                self.logger.info(str(post_fix))

                betas = {}
                for module_name, module in self.model.named_modules():
                    with torch.no_grad():

                        if isinstance(module, (FrequencyLayer, FrequencyLayer_Wavelet)):
                            beta = module.sqrt_beta **2

                            beta_norm = torch.norm(beta, p=2)
                            beta_max = torch.max(beta)
                            betas[f'params/{module_name}/beta_norm'] = beta_norm.item()
                            betas[f'params/{module_name}/beta_max'] = beta_max.item()
                        elif isinstance(module, MultiHeadFourierAttention):
                            beta = module.sqrt_beta ** 2

                            for head in range(beta.shape[1]):
                                head_beta = beta[0, head]  # shape: (1, seq_len)
                                head_norm = torch.norm(head_beta, p=2)
                                head_max = torch.max(head_beta)
                                betas[f'params/{module_name}/beta_norm_head_{head}'] = head_norm.item()
                                betas[f'params/{module_name}/beta_max_head_{head}'] = head_max.item()

                            grad = 2 * module.sqrt_beta * module.sqrt_beta.grad  # chain rule again
                            for head in range(grad.shape[1]):
                                head_grad = grad[0, head]
                                head_grad_norm = torch.norm(head_grad, p=2)
                                head_grad_max = torch.max(head_grad)
                                betas[f'grads/{module_name}/beta_grad_norm_head_{head}'] = head_grad_norm.item()
                                betas[f'grads/{module_name}/beta_grad_max_head_{head}'] = head_grad_max.item()

                if self.wandb:
                    wandb.log({
                        "train/rec_loss": rec_loss / len(rec_data_iter),
                        **betas,
                    }, step=epoch)

        else:
            with torch.no_grad():
                self.model.eval()

                all_pred_lists = []
                all_answer_lists = []
                rating_pred_stack = []

                for i, batch in rec_data_iter:
                    batch = tuple(t.to(self.device) for t in batch)
                    user_ids, input_ids, answers, _, _ = batch


                    recommend_output = self.model.predict(input_ids, user_ids, save=save)

                    recommend_output = recommend_output[:, -1, :]

                    rating_pred = self.predict_full(recommend_output)

                    batch_user_index = user_ids.cpu().numpy()
                    try:
                        mask_np = (self.args.train_matrix[batch_user_index].toarray() > 0)
                        mask = torch.from_numpy(mask_np).to(self.device)  # [batch × num_items]
                        rating_pred.masked_fill_(mask, 0)
                    except Exception as e:  # e.g., for bert4rec
                        rating_pred = rating_pred[:, :-1]
                        mask_np = (self.args.train_matrix[batch_user_index].toarray() > 0)
                        mask = torch.from_numpy(mask_np).to(self.device)
                        rating_pred.masked_fill_(mask, 0)
                        print(f"Warning: {e}")

                    if save and not spectral:
                        rating_pred_stack.extend(rating_pred.cpu().numpy())

                    topk_vals, topk_idx = torch.topk(rating_pred, k=20, dim=1)

                    all_pred_lists.append(topk_idx.cpu().numpy())
                    all_answer_lists.append(answers.cpu().numpy())

                pred_list = np.concatenate(all_pred_lists, axis=0)
                answer_list = np.concatenate(all_answer_lists, axis=0)

                # Write output test predictions if needed
                if save and not spectral:
                    file_path = os.path.join(self.args.save_path, self.args.data_name, f'{self.args.run_name}.json')
                    directory = os.path.join(self.args.save_path, self.args.data_name)
                    if not os.path.exists(directory):
                        os.makedirs(directory)

                    flat_scores = np.vstack(rating_pred_stack)
                    write_output(flat_scores, answer_list, file_path)

                scores, scores_string = self.get_full_sort_score(epoch, answer_list, pred_list)

                scores_dict = {
                    "val/HR@5": scores[0],
                    "val/NDCG@5": scores[1],
                    "val/HR@10": scores[2],
                    "val/NDCG@10": scores[3],
                    "val/HR@20": scores[4],
                    "val/NDCG@20": scores[5],
                }

                if self.wandb:
                    wandb.log(scores_dict, step=epoch)
                    return scores, scores_string

class FICLRecTrainer(Trainer):
    def __init__(self, model, train_dataloader, cluster_dataloader, eval_dataloader, test_dataloader, args, logger, wandb=True):
        super(FICLRecTrainer, self).__init__(model, train_dataloader, eval_dataloader, test_dataloader, args, logger, wandb)
        self.cluster_dataloader = cluster_dataloader
        # self.clusters_t should be initialized before training

    def train(self, epoch):
        return self.iteration(epoch,
                              dataloader=self.train_dataloader,
                              cluster_dataloader=self.cluster_dataloader,
                              full_sort=False,
                              train=True)

    def valid(self, epoch):
        self.args.train_matrix = self.args.valid_rating_matrix
        return self.iteration(epoch,
                              dataloader=self.eval_dataloader,
                              cluster_dataloader=None,
                              full_sort=True,
                              train=False)

    def test(self, epoch, save=False, spectral=False):
        self.args.train_matrix = self.args.test_rating_matrix
        return self.iteration(epoch,
                              dataloader=self.test_dataloader,
                              cluster_dataloader=None,
                              full_sort=True,
                              train=False,
                              save=save,
                              spectral=spectral)

    def iteration(self, epoch, dataloader, cluster_dataloader=None, full_sort=True, train=True, save=False, spectral=False):
        import gc
        # Contrastive clustering step
        if train and self.args.cl_mode in ['hl','l']:
            print("Preparing Clustering:")
            self.model.eval()
            feats_list = []
            for _, batch in tqdm.tqdm(enumerate(cluster_dataloader), total=len(cluster_dataloader)):
                batch = tuple(t.to(self.device) for t in batch)
                _, subsequence, _, _, _ = batch
                feats = self.model(subsequence)[:, -1, :].detach().cpu().numpy()
                feats_list.append(feats)
            feats = np.vstack(feats_list)
            for i, clusters in enumerate(self.clusters_t):
                for j, cluster in enumerate(clusters):
                    cluster.train(feats)
                    self.clusters_t[i][j] = cluster
            del feats; import gc; gc.collect()

        if train:
            print("Performing Rec model Training:")
            self.model.train()
            rec_loss_acc = 0.0
            icl_loss_acc = 0.0
            joint_loss_acc = 0.0
            for _, batch in tqdm.tqdm(enumerate(dataloader), total=len(dataloader)):
                print(f"Batch {_} of {len(dataloader)}")
                batch = tuple(t.to(self.device) for t in batch)
                _, seq1, targ1, seq2, _ = batch
                # prediction loss
                out1 = self.model(seq1)
                logits = self.predict_full(out1[:, -1, :])
                rec_loss = torch.nn.CrossEntropyLoss()(logits, targ1[:])
                # contrastive loss
                # ci1 = self.model(seq1)
                # ci2 = self.model(seq2)
                # hf = self.high_freq_loss([ci1, ci2], targ1) if self.args.cl_mode in ['h','hl'] else 0.0
                # lf = self.low_freq_loss([ci1, ci2], self.clusters_t[0]) if self.args.cl_mode in ['l','hl'] else 0.0
                # icl = self.args.alpha * hf + self.args.beta * lf
                # joint = self.args.rec_weight * rec_loss + icl
                self.optim.zero_grad()
                # joint.backward()
                self.optim.step()
                rec_loss_acc += rec_loss.item()
                # icl_loss_acc += (icl.item() if not isinstance(icl, float) else icl)
                # joint_loss_acc += joint.item()
            n = len(dataloader)
            metrics = {
                "train/rec_loss": rec_loss_acc / n,
                # "train/icl_loss": icl_loss_acc / n,
                # "train/joint_loss": joint_loss_acc / n
            }
            if self.wandb:
                wandb.log(metrics, step=epoch)
            return metrics
        else:
            if full_sort:
                return self._eval_full_sort(epoch, dataloader, save=False)
            else:
                return self._eval_sampled(epoch, dataloader)

    def _eval_full_sort(self, epoch, dataloader, save):
        all_preds, all_ans = [], []
        for _, batch in tqdm.tqdm(enumerate(dataloader), total=len(dataloader)):
            batch = tuple(t.to(self.device) for t in batch)
            uids, seq, targ, ans = batch[0], batch[1], batch[2], batch[3]
            out = self.model(seq)[:, -1, :]
            rating_pred = self.predict_full(out)
            mask = (self.args.train_matrix[uids.cpu().numpy()].toarray() > 0)
            rating_pred.masked_fill_(torch.from_numpy(mask).to(self.device), 0)
            topk = torch.topk(rating_pred, k=20, dim=1)[1].cpu().numpy()
            all_preds.append(topk)
            all_ans.append(ans.cpu().numpy())
        preds = np.concatenate(all_preds, axis=0)
        answ = np.concatenate(all_ans, axis=0)
        return self.get_full_sort_score(epoch, answ, preds)

    def _eval_sampled(self, epoch, dataloader):
        all_logits = []
        for _, batch in tqdm.tqdm(enumerate(dataloader), total=len(dataloader)):
            batch = tuple(t.to(self.device) for t in batch)
            _, seq, targ, neg, sem = batch
            out = self.model.finetune(seq)[:, -1, :]
            test_items = torch.cat((targ, sem), dim=1)
            logits = self.model.predict_sample(out, test_items)
            all_logits.append(logits.cpu().numpy())
        logits = np.vstack(all_logits)
        return self.get_sample_scores(epoch, logits)
