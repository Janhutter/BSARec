import os
import sys
import gdown
from tqdm import tqdm


def process_taobao():
    # download taobao dataset https://drive.usercontent.google.com/download?id=1oZJPdPtgT-yTVheY3CiK_7DXyURaOo26&export=download&authuser=0
    file_id = '1oZJPdPtgT-yTVheY3CiK_7DXyURaOo26'
    os.system(f'gdown {file_id} -O taobao.tar.gz')
    os.system('tar -zxvf taobao.tar.gz')
    os.system('rm taobao.tar.gz')

    data = {'label': [], 'user_hash': [], 'item_hash': [], 'item_cate': [], 'operation_time': [],
            'item_history_sequence': [], 'item_cate_history_sequence': [], 'time_history_sequence': []}
    histories = set()
    with open('taobao/test_data', 'r') as f:
        for i, line in enumerate(f):
            line = line.strip().split('\t')
            if line[5] in histories:
                continue

            histories.add(line[5])
            data['label'].append(line[0])
            data['user_hash'].append(line[1])
            data['item_hash'].append(line[2])
            data['item_cate'].append(line[3])
            data['operation_time'].append(line[4])

            history = line[5].split(',')
            if len(history) > 50:
                history = history[-50:]
            history.append(line[2])
            category = line[6].split(',')
            category.append(line[3])
            data['item_history_sequence'].append(history)
            data['item_cate_history_sequence'].append(category)
            data['time_history_sequence'].append(line[7])

    # select 10% random samples
    # sample_size = int(len(data['user_hash']) * 0.1)
    # import random
    # indices = random.sample(range(len(data['user_hash'])), sample_size)
    # data = {key: [data[key][i] for i in indices] for key in data}
    print(len(histories))
    with open('../taobao.txt', 'w') as f:
        for i in tqdm(range(len(data['user_hash']))):
            f.write(f"{data['user_hash'][i]} {' '.join(data['item_history_sequence'][i])}\n")

    mapping = {}
    for item, category in zip(data['item_hash'], data['item_cate']):
        if item not in mapping:
            mapping[item] = category

    with open('../taobao_item_category_map.txt', 'w') as f:
        for item, category in mapping.items():
            f.write(f"{item} {category}\n")

if __name__ == '__main__':
    process_taobao()