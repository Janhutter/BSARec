import tqdm
import numpy as np
import torch
import os
from scipy.sparse import csr_matrix
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import random

class RecDataset(Dataset):
    def __init__(self, args, user_seq, test_neg_items=None, data_type='train'):
        self.args = args
        self.user_seq = []
        self.max_len = args.max_seq_length
        self.user_ids = []
        self.contrastive_learning = args.model_type.lower() in ['fearec', 'duorec']
        self.data_type = data_type
        self.padding = args.padding.lower()


        if self.data_type=='train':
            for user, seq in enumerate(user_seq):
                input_ids = seq[-(self.max_len + 2):-2]
                for i in range(5, len(input_ids)):
                    self.user_seq.append(input_ids[:i + 1])
                    self.user_ids.append(user)
        elif self.data_type=='valid':
            for sequence in user_seq:
                self.user_seq.append(sequence[:-1])
        else:
            self.user_seq = user_seq

        self.test_neg_items = test_neg_items

        if self.contrastive_learning and self.data_type=='train':
            if os.path.exists(args.same_target_path):
                self.same_target_index = np.load(args.same_target_path, allow_pickle=True)
            else:
                print("Start making same_target_index for contrastive learning")
                self.same_target_index = self.get_same_target_index()
                # self.same_target_index = np.array(self.same_target_index)
                np.save(args.same_target_path, np.array(self.same_target_index, dtype=object))

    #     def get_same_target_index(self):
    #       num_items = max([max(v) for v in self.user_seq]) + 2
    #       same_target_index = [[] for _ in range(num_items)]
        
    #       user_seq = self.user_seq[:]
    #       tmp_user_seq = []
    #       for i in tqdm.tqdm(range(1, num_items)):
    #           for j in range(len(user_seq)):
    #               if user_seq[j][-1] == i:
    #                   same_target_index[i].append(user_seq[j])
    #               else:
    #                   tmp_user_seq.append(user_seq[j])
    #           user_seq = tmp_user_seq
    #           tmp_user_seq = [] 

    #       return same_target_index
    def apply_padding(self, user_seq_list):
        """
        Pad each sequence in user_seq_list so that its length >= max_len + 2.
        Padding is applied based on seq[:-2] to avoid using the last two items.
        Supports modes: 'zero', 'cyclic', 'reflect', 'mirror'.
        """
        target_length = self.max_len + 2
        padded = []

        for seq in user_seq_list:
            full_seq = np.array(seq, dtype=int)       # Full sequence to be padded at the end

            while len(full_seq) < target_length:
                pad_size = target_length - len(full_seq)

                if self.padding == 'zero':
                    pad = np.pad(full_seq, (pad_size, 0), mode='constant', constant_values=0)

                elif self.padding in ('reflect', 'mirror'):
                    mode = 'reflect' if self.padding == 'reflect' else 'symmetric'
                    pad = np.pad(full_seq, (pad_size, 0), mode=mode)

                elif self.padding == 'cyclic':
                    if len(full_seq) == 0:
                        raise ValueError("Cannot apply cyclic padding to empty base sequence")
                    repeats = int(np.ceil(pad_size / len(full_seq)))
                    cycle = np.tile(full_seq, repeats)
                    pad = np.concatenate([cycle[-pad_size:], full_seq])

                else:
                    raise ValueError(f"Unknown padding mode: {self.padding}")
                
                full_seq = pad
            arr = full_seq[-target_length:]  # Keep the last max_len + 2 items

            padded.append(arr.tolist())

        return padded


    def get_same_target_index(self):
        # 1) figure out how many distinct items there are
        num_items = max([max(v) for v in self.user_seq]) + 2
        same_target_index = [[] for _ in range(num_items)]

        # 2) one pass over self.user_seq
        for seq in self.user_seq:
            last_item = seq[-1]
            same_target_index[last_item].append(seq)

        return same_target_index

    def __len__(self):
        return len(self.user_seq)

    def __getitem__(self, index):
        items = self.user_seq[index]
        input_ids = items[:-1]
        answer = items[-1]

        seq_set = set(items)
        neg_answer = neg_sample(seq_set, self.args.item_size)

        input_ids = self.apply_padding([input_ids])[0] 

        input_ids = input_ids[-self.max_len:]
        assert len(input_ids) == self.max_len

        if self.data_type in ['valid', 'test']:
            cur_tensors = (
                torch.tensor(index, dtype=torch.long),  # user_id for testing
                torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(answer, dtype=torch.long),
                torch.zeros(0, dtype=torch.long), # not used
                torch.zeros(0, dtype=torch.long), # not used
            )

        elif self.contrastive_learning:
            sem_augs = self.same_target_index[answer]
            sem_aug = random.choice(sem_augs)
            keep_random = False
            for i in range(len(sem_augs)):
                if sem_augs[0] != sem_augs[i]:
                    keep_random = True

            while keep_random and sem_aug == items:
                sem_aug = random.choice(sem_augs)

            sem_aug = sem_aug[:-1]
            pad_len = self.max_len - len(sem_aug)
            sem_aug = [0] * pad_len + sem_aug
            sem_aug = sem_aug[-self.max_len:]
            assert len(sem_aug) == self.max_len

            cur_tensors = (
                torch.tensor(self.user_ids[index], dtype=torch.long),  # user_id for testing
                torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(answer, dtype=torch.long),
                torch.tensor(neg_answer, dtype=torch.long),
                torch.tensor(sem_aug, dtype=torch.long)
            )

        else:
            cur_tensors = (
                torch.tensor(self.user_ids[index], dtype=torch.long),  # user_id for testing
                torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(answer, dtype=torch.long),
                torch.tensor(neg_answer, dtype=torch.long),
                torch.zeros(0, dtype=torch.long), # not used
            )

        return cur_tensors


def neg_sample(item_set, item_size):
    item = random.randint(1, item_size - 1)
    while item in item_set:
        item = random.randint(1, item_size - 1)
    return item

def generate_rating_matrix_valid(user_seq, num_users, num_items):
    # three lists are used to construct sparse matrix
    row = []
    col = []
    data = []
    for user_id, item_list in enumerate(user_seq):
        for item in item_list[:-2]: #
            row.append(user_id)
            col.append(item)
            data.append(1)

    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    rating_matrix = csr_matrix((data, (row, col)), shape=(num_users, num_items))

    return rating_matrix

def generate_rating_matrix_test(user_seq, num_users, num_items):
    # three lists are used to construct sparse matrix
    row = []
    col = []
    data = []
    for user_id, item_list in enumerate(user_seq):
        for item in item_list[:-1]: #
            row.append(user_id)
            col.append(item)
            data.append(1)

    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    rating_matrix = csr_matrix((data, (row, col)), shape=(num_users, num_items))

    return rating_matrix

def get_rating_matrix(data_name, seq_dic, max_item):
    
    num_items = max_item + 1
    valid_rating_matrix = generate_rating_matrix_valid(seq_dic['user_seq'], seq_dic['num_users'], num_items)
    test_rating_matrix = generate_rating_matrix_test(seq_dic['user_seq'], seq_dic['num_users'], num_items)

    return valid_rating_matrix, test_rating_matrix

def get_user_seqs_and_max_item(data_file):
    lines = open(data_file).readlines()
    lines = lines[1:]
    user_seq = []
    item_set = set()
    for line in lines:
        user, items = line.strip().split('	', 1)
        items = items.split()
        items = [int(item) for item in items]
        user_seq.append(items)
        item_set = item_set | set(items)
    max_item = max(item_set)
    return user_seq, max_item

def get_user_seqs(data_file):
    lines = open(data_file).readlines()
    user_seq = []
    item_set = set()
    for line in lines:
        user, items = line.strip().split(' ', 1)
        items = items.split(' ')
        items = [int(item) for item in items]
        user_seq.append(items)
        item_set = item_set | set(items)
    max_item = max(item_set)
    num_users = len(lines)

    return user_seq, max_item, num_users

def get_seq_dic(args):

    args.data_file = args.data_dir + args.data_name + '.txt'
    user_seq, max_item, num_users = get_user_seqs(args.data_file)
    seq_dic = {'user_seq':user_seq, 'num_users':num_users }

    return seq_dic, max_item, num_users

def get_dataloder(args,seq_dic):
    

    # history length of validation and test is longer than training
    # the last item of the user history is used as label for test,
    # the one before this for validation
    # and the one before that as training label
    train_dataset = RecDataset(args, seq_dic['user_seq'], data_type='train')
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size, num_workers=args.num_workers)

    eval_batch_size = args.batch_size // 4 if args.batch_size > 4 else args.batch_size
    eval_dataset = RecDataset(args, seq_dic['user_seq'], data_type='valid')
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=eval_batch_size, num_workers=args.num_workers)

    test_batch_size = args.batch_size // 16 if args.batch_size > 16 else args.batch_size
    test_dataset = RecDataset(args, seq_dic['user_seq'], data_type='test')
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=test_batch_size, num_workers=args.num_workers)\
    
    cluster_dataset = RecDataset(args, seq_dic['user_seq'], data_type='train')
    cluster_sampler = SequentialSampler(cluster_dataset)
    cluster_dataloader = DataLoader(cluster_dataset, sampler=cluster_sampler, batch_size=args.batch_size, num_workers=args.num_workers)

    return train_dataloader, eval_dataloader, test_dataloader, cluster_dataloader

