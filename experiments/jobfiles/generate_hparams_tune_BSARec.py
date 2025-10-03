import random
hyperparameter_file = 'hparams_tune_BSARec.txt'

parameters = {
    # Basic sweepable hyperparameters
    'learning_rate': [0.01, 0.005],
    'batch_size': [1024],
    'epochs': [200],
    # if values below here are set as range, ensure to change hyperparameter generation logic accordingly
    'attention_probs_dropout_prob': [0.1],
    'hidden_dropout_prob': [0.1],
    'num_attention_heads': [1,2,4],
    # 'datasets': ['Beauty', 'Sports_and_Outdoors', 'Toys_and_Games', 'LastFM', 'ML-1M', 'Yelp'],
    'datasets': ['MIND', 'TaoBao'],
    # 'model_type': ['BSARec', 'BERT4Rec', 'Caser', 'SASRec', 'GRU4Rec', 'duoRec', 'FeaRec', 'SASRec', 'FMLPRec'],
    'model_type': ['BSARec'],


    # --- Model-specific hyperparameters (can be tuned later) ---
    'alpha': [0.1, 0.3, 0.5, 0.7, 0.9],
    'c': [1, 3, 5, 7, 9],
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
N_seeds = 1
seed_list = random.sample(range(0, 10000), N_seeds) 

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
                                        for alpha in parameters['alpha']:  
                                            for c in parameters['c']:
                                                name = f"{model}_{dataset}_lr_{lr}_bs_{bs}_ep_{epoch}_c_{c}_alpha_{alpha}_num_heads_{num_heads}"
                                                

                                                
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
                                                    f"--alpha {alpha} "
                                                    f"--c {c} "
                                                    f"--project BSARec_Tuning"
                                                    f"\n"
                                                )

                                                f.write(line)

                        