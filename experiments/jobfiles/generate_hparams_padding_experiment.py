import random
hyperparameter_file = 'hparams_padding_experiment.txt'

parameters = {
    # Basic sweepable hyperparameters
    'learning_rate': [0.001],
    'batch_size': [256],
    'epochs': [200],
    # if values below here are set as range, ensure to change hyperparameter generation logic accordingly
    'attention_probs_dropout_prob': [0.5],
    'hidden_dropout_prob': [0.5],
    'num_attention_heads': [1],
    # 'datasets': ['Beauty', 'Sports_and_Outdoors', 'Toys_and_Games', 'LastFM', 'ML-1M', 'Yelp'],
    'datasets': ['LastFM', 'ML-1M', 'Yelp'],
    # 'model_type': ['BSARec', 'BERT4Rec', 'Caser', 'SASRec', 'GRU4Rec', 'duoRec', 'FeaRec', 'SASRec', 'FMLPRec'],
    'model_type': ['FeaRec', 'SASRec', "FMLPRec"],
    'padding': ['reflect', 'zero', 'cyclic', 'mirror'],
    "no_padding": [False], 
    "flip_zero_padding": [True],
    'max_seq_length': [50],

    'mask_ratio': [0.2],
    'nh': [8],
    'nv': [4],
    'reg_weight': [1e-4],
    'tau': [1.0],
    'lmd': [0.1],
    'lmd_sem': [0.1],
    'ssl': ['us_x'],
    'sim': ['dot'],
    'spatial_ratio': [0.1],
    'global_ratio': [0.6],
    'fredom_type': ['us_x'],
    'fredom': ['True'],
    'gru_hidden_size': [64]
}

model_specific_args = {
    'mask_ratio': ['bert4rec'],
    'nh': ['caser'],
    'nv': ['caser'],
    'reg_weight': ['caser'],
    'tau': ['duorec', 'fearec'],
    'lmd': ['duorec', 'fearec'],
    'lmd_sem': ['duorec', 'fearec'],
    'ssl': ['duorec', 'fearec'],
    'sim': ['duorec', 'fearec'],
    'spatial_ratio': ['fearec'],
    'global_ratio': ['fearec'],
    'fredom_type': ['fearec'],
    'fredom': ['fearec'],
    'gru_hidden_size': ['gru4rec'],
}
# generate random seed

N_seeds = 3
seed_list = random.sample(range(0, 10000), N_seeds)
seed_list = [38, 39, 40, 41]

bsarec_alphas = {
    'Beauty': 0.7,
    'Sports_and_Outdoors': 0.3,
    'Toys_and_Games': 0.7,
    'Yelp': 0.7,
    'LastFM': 0.9,
    'ML-1M': 0.3,
    'taobao_small': 0.1,
    'MIND': 0.3,
}

bsarec_cs = {
    'Beauty': 5,
    'Sports_and_Outdoors': 5,
    'Toys_and_Games': 3,
    'Yelp': 3,
    'LastFM': 3,
    'ML-1M': 9,
    'taobao_small': 9,
    'MIND': 9,
}

bsarec_attention_heads = {
    'Beauty': 1,
    'Sports_and_Outdoors': 4,
    'Toys_and_Games': 1,
    'Yelp': 4,
    'LastFM': 1,
    'ML-1M': 4,
    'taobao_small': 4,
    'MIND': 1,
}
bsarec_lrs = {
    'Beauty': 5e-4,
    'Sports_and_Outdoors': 1e-3,
    'Toys_and_Games': 1e-3,
    'Yelp': 1e-3,
    'LastFM': 5e-4,
    'ML-1M': 1e-3,
    'taobao_small': 0.005,
    'MIND': 0.005,
}

additional_args = ''

with open(hyperparameter_file, 'w') as f:
    for dataset in parameters['datasets']:
        for model in parameters['model_type']:
            for lr in parameters['learning_rate']:
                for bs in parameters['batch_size']:
                    for epoch in parameters['epochs']:
                        for attn_dropout in parameters['attention_probs_dropout_prob']:
                            for hidden_dropout in parameters['hidden_dropout_prob']:
                                for num_heads in parameters['num_attention_heads']:
                                    for padding in parameters['padding']:
                                        for sequence_length in parameters['max_seq_length']:
                                                for seed in seed_list:
                                                    # Compose model-specific args
                                                    model_lower = model.lower()
                                                    additional_args = ""

                                                    if model == 'BSARec_Padding':
                                                        alpha = bsarec_alphas.get(dataset)
                                                        c = bsarec_cs.get(dataset)
                                                        num_heads = bsarec_attention_heads.get(dataset)
                                                        lr = bsarec_lrs.get(dataset)                                
                                                        additional_args = f"--c {c} --alpha {alpha}  "

                                                    for param, models in model_specific_args.items():
                                                        if model_lower in models:
                                                            values = parameters.get(param)
                                                            if values:
                                                                additional_args += f"--{param} {values[0]} "

                                                    name = f"{model}_{dataset}_lr_{lr}_bs_{bs}_ep_{epoch}_padding_type_{padding}_seed_{seed}"

                                                    line = (
                                                        f"--data_name {dataset} "
                                                        f"--train_name {name} "
                                                        f"--run_name {name} "
                                                        f"--lr {lr} "
                                                        f"--batch_size {bs} "
                                                        f"--epochs {epoch} "
                                                        f"--attention_probs_dropout_prob {attn_dropout} "
                                                        f"--hidden_dropout_prob {hidden_dropout} "
                                                        f"--num_attention_heads {num_heads} "
                                                        f"--model_type {model} "
                                                        f"--data_name {dataset} "
                                                        f"{additional_args.strip()} "
                                                        f"--save "
                                                        f"--save_path /scratch-shared/recsys/ "
                                                        f"--output_dir /scratch-shared/recsys/ "
                                                        f"--seed {seed} "
                                                        f"--project padding_experiment_final "
                                                        f"--padding {padding} "
                                                        f"--max_seq_length {sequence_length} "
                                                    )

                                                    f.write(line + '\n')

                                