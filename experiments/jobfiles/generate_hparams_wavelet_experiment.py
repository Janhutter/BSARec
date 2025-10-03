import random
hyperparameter_file = 'hparams_wavelet_experiment.txt'

parameters = {
    # Basic sweepable hyperparameters
    'learning_rate': [0.001],
    'batch_size': [8192],
    'epochs': [200],
    # if values below here are set as range, ensure to change hyperparameter generation logic accordingly
    'attention_probs_dropout_prob': [0.1],
    'hidden_dropout_prob': [0.1],
    'num_attention_heads': [1],
    # 'datasets': ['Beauty', 'Sports_and_Outdoors', 'Toys_and_Games', 'LastFM', 'ML-1M', 'Yelp'],
    'datasets': ['LastFM', 'ML-1M'],
    # 'model_type': ['BSARec', 'BERT4Rec', 'Caser', 'SASRec', 'GRU4Rec', 'duoRec', 'FeaRec', 'SASRec', 'FMLPRec'],
    'model_type': ['BSARec_Wavelet', 'BSARec_skip'],


    # --- Model-specific hyperparameters (can be tuned later) ---
    'wavelet_mode': ['symmetric', 'reflect', 'constant', 'periodic'],
    'wavelet': ['haar', 'sym4'],
    
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

bsarec_alphas = {
    'LastFM': 0.5,
    'ML-1M': 0.5
}

bsarec_cs = {
    'LastFM': 3,
    'ML-1M': 9
}

bsarec_attention_heads = {
    'LastFM': 1,
    'ML-1M': 4
}
bsarec_lrs = {
    'LastFM': 5e-3,
    'ML-1M': 1e-3
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
                                    for wavelet in parameters['wavelet']:
                                        for wavelet_mode in parameters['wavelet_mode']:
                                            for seed in seed_list:

                                                # Compose model-specific args
                                                model_lower = model.lower()
                                                additional_args = ""

                                                if model_lower == 'bsarec_wavelet':
                                                    alpha = bsarec_alphas.get(dataset)
                                                    c = bsarec_cs.get(dataset)
                                                    num_heads = bsarec_attention_heads.get(dataset)
                                                    lr = bsarec_lrs.get(dataset)                                
                                                    additional_args = f"--c {c} --alpha {alpha} "

                                                for param, models in model_specific_args.items():
                                                    if model_lower in models:
                                                        values = parameters.get(param)
                                                        if values:
                                                            additional_args += f"--{param} {values[0]} "

                                                name = f"{model}_{dataset}_lr_{lr}_bs_{bs}_ep_{epoch}_wavelet_{wavelet}_wavelet_mode_{wavelet_mode}_seed_{seed}"

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
                                                    f"--project Wavelet_Experiment "
                                                    f"--wavelet_mode {wavelet_mode} "
                                                    f"--wavelet {wavelet} "
                                                    f"\n"
                                                )

                                                f.write(line)

                            