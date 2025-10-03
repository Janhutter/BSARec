import os
import torch
from torch.profiler import profile, record_function, ProfilerActivity
import numpy as np

from model import MODEL_DICT
from trainers import Trainer, FICLRecTrainer
from utils import EarlyStopping, check_path, set_seed, parse_args, set_logger
from dataset import get_seq_dic, get_dataloder, get_rating_matrix
import wandb

def main():
    args = parse_args()
    log_path = os.path.join(args.output_dir, args.train_name + '.log')
    logger = set_logger(log_path)
    # init wandb

    dictionary = vars(args)
    dictionary['job_id'] = os.environ.get('SLURM_JOBID', 'local_run')
    wandb.init(
        project=args.project,
        name=args.run_name,
        config=vars(args)
    )

    set_seed(args.seed)
    check_path(args.output_dir)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    args.cuda_condition = torch.cuda.is_available() and not args.no_cuda

    seq_dic, max_item, num_users = get_seq_dic(args)
    args.item_size = max_item + 1
    args.num_users = num_users + 1

    args.checkpoint_path = os.path.join(args.output_dir, args.train_name + '.pt')
    args.same_target_path = os.path.join(args.data_dir, args.data_name+'_same_target' + str(args.max_seq_length) + '.npy')
    train_dataloader, eval_dataloader, test_dataloader, cluster_dataloader = get_dataloder(args,seq_dic)

    logger.info(str(args))
    model = MODEL_DICT[args.model_type.lower()](args=args)
    logger.info(model)

    # log model during training
    if args.wandb_log_model:
        wandb.watch(model, log_freq=args.wandb_log_freq_model)

    if args.model_type.lower() == 'ficlrec':
        trainer = FICLRecTrainer(model, train_dataloader, cluster_dataloader, eval_dataloader, test_dataloader, args, logger)
    else:
        trainer = Trainer(model, train_dataloader, eval_dataloader, test_dataloader, args, logger)


    args.valid_rating_matrix, args.test_rating_matrix = get_rating_matrix(args.data_name, seq_dic, max_item)

    if args.do_eval:
        if args.load_model is None:
            logger.info(f"No model input!")
            exit(0)
        else:
            args.checkpoint_path = os.path.join(args.output_dir, args.load_model + '.pt')
            trainer.load(args.checkpoint_path)

            logger.info(f"Load model from {args.checkpoint_path} for test!")
            scores, result_info = trainer.test(epoch=args.epochs - 1, save=args.save, spectral=args.spectral)
            args.checkpoint_path = os.path.join(args.output_dir, args.train_name + '.pt')
            # torch.save(trainer.model.state_dict(), args.checkpoint_path)

    else:
        early_stopping = EarlyStopping(args.checkpoint_path, logger=logger, patience=args.patience, verbose=True)

        for epoch in range(args.epochs):
            trainer.train(epoch)
            scores, _ = trainer.valid(epoch)

            # evaluate on MRR
            early_stopping(np.array(scores[-1:]), trainer.model)
            if early_stopping.early_stop:
                logger.info("Early stopping")
                break

        logger.info("---------------Test Score---------------")
        torch.cuda.empty_cache()

        trainer.model.load_state_dict(torch.load(args.checkpoint_path))
        scores, result_info = trainer.test(epoch=args.epochs - 1, save=args.save, spectral=args.spectral)

    logger.info(args.train_name)
    logger.info(result_info)

    scores_dict = {
                "test/HR@5": scores[0],
                "test/NDCG@5": scores[1],
                "test/HR@10": scores[2],
                "test/NDCG@10": scores[3],
                "test/HR@20": scores[4],
                "test/NDCG@20": scores[5],
    }
    # log scores to wandb
    wandb.log(scores_dict)

    # finish wandb run
    wandb.finish()


main()
