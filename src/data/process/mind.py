
import pandas as pd
import numpy as np
import os

def process_mind():
    data = {}

    train_link = 'https://recodatasets.z20.web.core.windows.net/newsrec/MINDsmall_train.zip'
    test_link = 'https://recodatasets.z20.web.core.windows.net/newsrec/MINDsmall_dev.zip'

    os.system(f'wget {train_link} -O MINDsmall_train.zip')
    os.system(f'wget {test_link} -O MINDsmall_dev.zip')

    # create folders and unzip into them
    os.makedirs('MINDsmall_train', exist_ok=True)
    os.makedirs('MINDsmall_dev', exist_ok=True)
    os.system('unzip -o MINDsmall_train.zip -d MINDsmall_train')
    os.system('unzip -o MINDsmall_dev.zip -d MINDsmall_dev')

    # remove the zip files
    os.system('rm MINDsmall_train.zip')
    os.system('rm MINDsmall_dev.zip')

    loc = 'MINDsmall_dev/behaviors.tsv'
    loc2 = 'MINDsmall_train/behaviors.tsv'

    df = pd.read_csv(loc, sep='\t', header=None, names=['impression_id', 'user_id', 'time', 'history', 'impressions'])
    df_2 = pd.read_csv(loc2, sep='\t', header=None, names=['impression_id', 'user_id', 'time', 'history', 'impressions'])
    df['time'] = pd.to_datetime(df['time'], format='%m/%d/%Y %I:%M:%S %p')
    df_2['time'] = pd.to_datetime(df_2['time'], format='%m/%d/%Y %I:%M:%S %p')

    df = pd.concat([df, df_2], ignore_index=True)

    df = df.sort_values(by='time', ascending=True)


    for i, row in df.iterrows():
        user_id = row['user_id']
        if str(row['history']) == 'nan':
            continue
        user_id = user_id[1:]  # remove the first character which is 'U'

        history = row['impressions'].split(' ')
        if user_id not in data:
            data[user_id] = [h[1:] for h in row['history'].split(' ')]
        data[user_id].extend([h.split('-1')[0][1:] for h in history if '-0' not in h])

    # limit user history to 50 items
    for user_id in data:
        if len(data[user_id]) > 50:
            data[user_id] = data[user_id][-50:]

    with open('../MIND.txt', 'w') as f:
        for user_id, history in data.items():
            f.write(f"{user_id} {' '.join(history)}\n")

    mapping_file_1 = 'MINDsmall_train/news.tsv'
    mapping_file_2 = 'MINDsmall_dev/news.tsv'

    df_mapping_1 = pd.read_csv(mapping_file_1, sep='\t', header=None)
    df_mapping_2 = pd.read_csv(mapping_file_2, sep='\t', header=None)
    df_mapping = pd.concat([df_mapping_1, df_mapping_2], ignore_index=True)
    print(df_mapping.head())
    df_mapping = df_mapping[[0, 1]]
    df_mapping[0] = df_mapping[0].str[1:]  # remove the first character which is 'N'

    df_mapping = df_mapping.drop_duplicates(subset=0)
    df_mapping = df_mapping.set_index(0)
    df_mapping = df_mapping.sort_index()
    with open('../MIND_item_category_map.txt', 'w') as f:
        for item, category in df_mapping.iterrows():
            f.write(f"{item} {category[1]}\n")

if __name__ == '__main__':
    process_mind()