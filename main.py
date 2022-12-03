import os
import torch
import argparse
import pickle
from pytorch_lightning import seed_everything
from model.cifar10.cnn import CNN as CIFAR_CNN
from model.cifar10.mlp import MLP as CIFAR_MLP
from model.mnist.cnn import CNN as MNIST_CNN
from model.mnist.mlp import MLP as MNIST_MLP
from server import Server
from client import Client
from util import create_model
import wandb
from dataset.datasource import DataLoaders
from torchmetrics import MetricCollection, Accuracy, Precision, Recall
from torchmetrics import F1Score as F1
from pathlib import Path
from util import *
from util_dev import *
import random

from datetime import datetime

models = {
    'cifar10': {
        'cnn': CIFAR_CNN,
        'mlp': CIFAR_MLP
    },
    'mnist': {
        'cnn': MNIST_CNN,
        'mlp': MNIST_MLP
    }
}

def set_seed(seed):
    seed_everything(seed, workers=True)
    os.environ['PYTHONHASHSEED']=str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', help="mnist|cifar10",
                        type=str, default="mnist")
    parser.add_argument('--arch', type=str, default='cnn', help='cnn|mlp')
    parser.add_argument('--dataset_mode', type=str,
                        default='non-iid', help='non-iid|iid')
    parser.add_argument('--rate_unbalance', type=float, default=1.0)
    parser.add_argument('--num_clients', type=int, default=12)
    parser.add_argument('--rounds', type=int, default=15)
    parser.add_argument('--prune_step', type=float, default=0.2)
    parser.add_argument('--prune_threshold', type=float, default=0.8)
    parser.add_argument('--frac_clients_per_round', type=float, default=1.0)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--n_samples', type=int, default=20)
    parser.add_argument('--n_class', type=int, default=3)
    parser.add_argument('--optimizer', type=str, default="Adam", help="SGD|Adam")
    
    parser.add_argument('--train_verbose', type=bool, default=False)
    parser.add_argument('--test_verbose', type=bool, default=False)
    parser.add_argument('--prune_verbose', type=bool, default=False)
    parser.add_argument('--seed', type=int, default=40)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--num_dataloader_workers', type=int, default=0)
    

    parser.add_argument('--save_freq', type=int, default=10)
    parser.add_argument('--log_dir', type=str, default="./logs")

    # Run Type
    parser.add_argument('--CELL', type=int, default=0)
    parser.add_argument('--standalone', type=int, default=0)
    parser.add_argument('--fedavg_no_prune', type=int, default=0)

    # Run Type - overlapping
    parser.add_argument('--overlapping_prune', type=int, default=0, help='prune based on overlapping ratio')
    parser.add_argument('--prune_by_top', type=int, default=0, help='prune based low prune_threshold (target_sparsity) overlapping ratio')
    parser.add_argument('--prune_by_low', type=int, default=0, help='prune based top prune_threshold (target_sparsity) overlapping ratio with l1 pruning')
    parser.add_argument('--top_overlapping_threshold', type=float, default=0.5, help='how much overlapping region to consider, usually equal to prune_threshold. CHANGE noise_targeted_percent TOO!')
    
    # for CELL
    parser.add_argument('--eita', type=float, default=0.5,
                        help="accuracy threshold")
    parser.add_argument('--alpha', type=float, default=0.5,
                        help="accuracy reduction factor")

    # for POLL
    # parser.add_argument('--POLL', type=int, default=0)
    # parser.add_argument('--diff_freq', type=int, default=2)

    # for federated malicious
    parser.add_argument('--noise_variance', type=float, default=1, help="noise variance of the Gaussian Noise by malicious clients")
    parser.add_argument('--noise_targeted_percent', type=float, default=0.5, help="percent of weights on TOP targeted positions to introduce noise, usually equal to top_overlapping_threshold. low positions will be calculated as 1 - noise_targeted_percent")
    parser.add_argument('--n_malicious', type=int, default=0, help="number of malicious nodes in the network")

    # for overlapping prune
    parser.add_argument('--deterministic_data_shards', type=int, default=0, help="to verify rewarding mechanism based on common labels")
    # parser.add_argument('--overlapping_prune', type=int, default=1, help='prune based low prune_threshold (target_sparsity) overlapping ratio')
    # parser.add_argument('--noise_targeted_percent', type=float, default=0.2, help="percent of weights on TOP targeted positions to introduce noise. low positions will be calculated as 1 - noise_targeted_percent")

    # for debug
    parser.add_argument('--save_data_loaders', type=int, default=0)
    parser.add_argument('--save_intermediate_models', type=int, default=0)
    parser.add_argument('--save_full_local_models', type=int, default=0)
    parser.add_argument('--save_global_models', type=int, default=1)

    parser.add_argument('--wandb_username', type=str, default=None)
    parser.add_argument('--wandb_project', type=str, default="dummy")
    parser.add_argument('--run_note', type=str, default="")

    args = parser.parse_args()

    set_seed(args.seed)

    args.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device {args.device}")
    model = create_model(cls=models[args.dataset]
                         [args.arch], device=args.device)

    exe_date_time = datetime.now().strftime("%m%d%Y_%H%M%S")

    log_root_name = f"{exe_date_time}_SEED_{args.seed}_NOISE_PERCENT_{args.noise_targeted_percent}_VAR_{args.noise_variance}"

    try:
        # on Google Drive
        import google.colab
        args.log_dir = f"/content/drive/MyDrive/POLL_BASE/{log_root_name}"
    except:
        # local
        args.log_dir = f"{args.log_dir}/{log_root_name}"
    os.makedirs(args.log_dir)
    print(f"Model weights saved at {args.log_dir}.")

    model_save_path = f"{args.log_dir}/models_weights/globals_0"
    Path(model_save_path).mkdir(parents=True, exist_ok=True)
    trainable_model_weights = get_trainable_model_weights(model)
    with open(f"{model_save_path}/R0.pkl", 'wb') as f:
        pickle.dump(trainable_model_weights, f)

    train_loaders, test_loaders, user_labels, global_test_loader = DataLoaders(num_users=args.num_clients,
                                              dataset_name=args.dataset,
                                              n_class=args.n_class,
                                              nsamples=args.n_samples,
                                              log_dirpath=args.log_dir,
                                              mode=args.dataset_mode,
                                              batch_size=args.batch_size,
                                              rate_unbalance=args.rate_unbalance,
                                              num_workers=args.num_dataloader_workers,
                                              deterministic_sharding=args.deterministic_data_shards
                                              )
    clients = []
    n_malicious = args.n_malicious
    Path(f"{args.log_dir}/data_loaders/").mkdir(parents=True, exist_ok=True)
    for i in range(args.num_clients):
        malicious = True if args.num_clients - i <= n_malicious else False
        client = Client(i + 1, args, malicious, train_loaders[i], test_loaders[i], user_labels[i], global_test_loader)
        clients.append(client)
        # save data loader as validation set of the client
        if args.save_data_loaders:
            L_or_M = "M" if malicious else "L"
            torch.save(train_loaders[i], f"{args.log_dir}/data_loaders/{L_or_M}_{user_labels[i]}_{i+1}.dataloader")

    if args.standalone:
        run_name = "STANDALONE" # Pure Centralized
    if args.fedavg_no_prune:
        run_name = "FEDAVG_NO_PRUNE" # Pure FedAvg without Pruning
    if args.CELL:
        run_name = "CELL"
    if args.overlapping_prune and args.prune_by_top:
        run_name = "TOP_OVERLAPPING_PRUNE"
    if args.overlapping_prune and args.prune_by_low:
        run_name = "LOW_OVERLAPPING_PRUNE"
    # if args.overlapping:
    #     run_name = "OVERLAPPING"
    # POLL, standalone_poll, standalone_cell, standalone_speed

    wandb.login()
    wandb.init(project=args.wandb_project, entity=args.wandb_username)
    wandb.run.name = f"{run_name}_samples_{args.n_samples}_n_clients_{args.num_clients}_mali_{args.n_malicious}_optim_{args.optimizer}_seed_{args.seed}_{args.run_note}_{exe_date_time}"
    wandb.config.update(args)


    server = Server(args, model, clients, global_test_loader)

    for comm_round in range(1, args.rounds+1):
        server.update(comm_round)

