import wandb
from typing import List, Dict, Tuple
import torch.nn.utils.prune as prune
import numpy as np
import random
import os
from tabulate import tabulate
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import Module
from util import get_prune_params, super_prune, fed_avg, l1_prune, create_model, copy_model, get_prune_summary
from util import test as util_test
from util import *
from util_dev import *

class Server():
    """
        Central Server
    """

    def __init__(
        self,
        args,
        model,
        clients,
        global_test_loader
    ):
        super().__init__()
        self.clients = clients
        self.num_clients = len(self.clients)
        self.args = args
        self.model = model
        self.init_model = copy_model(model, self.args.device)

        self.curr_prune_step = 0.00

        self.global_test_loader = global_test_loader

    def aggr(
        self,
        models,
        clients,
        *args,
        **kwargs
    ):
        weights_per_client = np.array(
            [client.num_data for client in clients], dtype=np.float32)
        weights_per_client /= np.sum(weights_per_client)

        aggr_model = fed_avg(
            models=models,
            weights=weights_per_client,
            device=self.args.device
        )

        if self.args.CELL:
            pruned_percent = get_prune_summary(aggr_model, name='weight')['global']
            # pruned by the earlier zeros in the model
            l1_prune(aggr_model, amount=pruned_percent, name='weight')

        if self.args.overlapping_prune:
            # apply mask object to aggr_model. Otherwise won't work in lowOverlappingPrune()
            l1_prune(model=aggr_model,
                    amount=0.00,
                    name='weight',
                    verbose=False)
        
        return aggr_model

    def update(
        self,
        comm_round,
        *args,
        **kwargs
    ):
        """
            Interface to server and clients
        """
        print('-----------------------------', flush=True)
        print(
            f'| Communication Round: {comm_round}  | ', flush=True)
        print('-----------------------------', flush=True)

        client_idxs = np.random.choice(
            self.num_clients, int(
                self.args.frac_clients_per_round*self.num_clients),
            replace=False,
        )
        clients = [self.clients[i] for i in client_idxs]

        # for the ease of debugging overlapping labels
        clients.sort(key=lambda x: x.idx)

        # upload model to selected clients
        self.upload(clients)

        prune_rate = get_prune_summary(model=self.model,
                                       name='weight')['global']
        print('Global model prune percentage before starting: {}'.format(prune_rate))
        wandb.log({f"global_model_pruned_percentage_at_the_beginning": prune_rate, "comm_round": comm_round})

        # call training loop on all clients
        for client in clients:
            if self.args.standalone:
                client.update_standalone()
            if self.args.fedavg_no_prune:
                client.update_fedavg_no_prune(comm_round)
            if self.args.CELL:
                client.update_CELL(comm_round)
            if self.args.overlapping_prune:
                client.update_overlapping_prune(comm_round)
                
        
        if self.args.standalone:
            import sys
            sys.exit()
        
        # download models from selected clients
        models, accs, last_local_model_paths = self.download(clients)

        if self.args.CELL or self.args.overlapping_prune:
            model_type = "ticket"
        if self.args.fedavg_no_prune:
            model_type = "local"

        avg_accuracy = np.mean(accs, axis=0, dtype=np.float32)
        print('-----------------------------', flush=True)
        print(f'| Average {model_type.title()} Model Accuracy on Local Test Sets: {avg_accuracy}  | ', flush=True)
        print('-----------------------------', flush=True)
        wandb.log({f"avg_{model_type}_model_local_acc": avg_accuracy, "comm_round": comm_round})

        # compute average-model
        aggr_model = self.aggr(models, clients)

        # test UNpruned global model on entire test set
        global_test_acc = util_test(aggr_model,
                               self.global_test_loader,
                               self.args.device,
                               self.args.test_verbose)['Accuracy'][0]
        print('-----------------------------', flush=True)
        print(f'| Un-pruned Global Model on Global Test Set Accuracy at Round {comm_round} : {global_test_acc}  | ', flush=True)
        print('-----------------------------', flush=True)
        wandb.log({"unpruned_global_model_global_acc": global_test_acc, "comm_round": comm_round})

         # test UNpruned global model on each local test set
        for client in self.clients:
            global_model_local_set_acc = client.eval(self.model)["Accuracy"][0]
            print(f"Un-pruned Global Model on Client {client.idx} Local Test Set Accuracy at Round {comm_round} : {global_model_local_set_acc}")
            wandb.log({f"{client.idx}_unpruned_global_model_local_acc": global_model_local_set_acc, "comm_round": comm_round})

        layer_TO_if_pruned = [False]
        if self.args.overlapping_prune:
            if self.args.prune_by_low:
                # get low overlappings
                layer_TO_if_pruned, layer_TO_pruned_percentage = prune_by_low_overlap(aggr_model, last_local_model_paths, self.args.prune_threshold, self.args.device)
            if self.args.prune_by_top:
                layer_TO_if_pruned, layer_TO_pruned_percentage = prune_by_top_overlap_l1(aggr_model, last_local_model_paths, self.args.top_overlapping_threshold, self.args.prune_threshold)
            # log pruned amount of each layer
            for layer, pruned_percentage in layer_TO_pruned_percentage.items():
                print(f"Pruned percentage of {layer}: {pruned_percentage:.2%}")
                wandb.log({f"{layer}_pruned_percentage": pruned_percentage, "comm_round": comm_round})

        # save global model
        if self.args.save_global_models:
            model_save_path = f"{self.args.log_dir}/models_weights/globals_0"
            trainable_model_weights = get_trainable_model_weights(aggr_model)
            with open(f"{model_save_path}/R{comm_round}.pkl", 'wb') as f:
                pickle.dump(trainable_model_weights, f)

        if not self.args.overlapping_prune:
            # copy aggregated-model's params to self.model (keep buffer same)
            source_params = dict(aggr_model.named_parameters())
            for name, param in self.model.named_parameters():
                param.data.copy_(source_params[name])

        # test PRUNED global model on entire test set
        global_test_acc = util_test(self.model,
                               self.global_test_loader,
                               self.args.device,
                               self.args.test_verbose)['Accuracy'][0]
        print('-----------------------------', flush=True)
        print(f'| Pruned Global Model on Global Test Set Accuracy at Round {comm_round} : {global_test_acc}  | ', flush=True)
        print('-----------------------------', flush=True)
        wandb.log({"pruned_global_model_global_acc": global_test_acc, "comm_round": comm_round})

        # test PRUNED global model on each local test set
        for client in self.clients:
            global_model_local_set_acc = client.eval(self.model)["Accuracy"][0]
            print(f"Pruned Global Model on Client {client.idx} Local Test Set Accuracy at Round {comm_round} : {global_model_local_set_acc}")
            wandb.log({f"{client.idx}_pruned_global_model_local_acc": global_model_local_set_acc, "comm_round": comm_round})

        # log global model prune percentage
        if self.args.CELL:
            global_prune_rate = get_prune_summary(model=self.model,
                                        name='weight')['global']
            print(f'Global model prune percentage at the end: {global_prune_rate}')
            wandb.log({f"global_model_prune_percentage": global_prune_rate, "comm_round": comm_round})

        if self.args.overlapping_prune and True in layer_TO_if_pruned.values():
            # reinit
            self.model = aggr_model
            source_params = dict(self.init_model.named_parameters())
            for name, param in self.model.named_parameters():
                param.data.copy_(source_params[name].data)
            print("Params reinitialized.")
        else:
            print("Params NOT reinitialized.")

    def download(
        self,
        clients,
        *args,
        **kwargs
    ):
        # downloaded models are non pruned (taken care of in fed-avg)
        uploads = [client.upload() for client in clients]
        models = [upload["model"] for upload in uploads]
        accs = [upload["acc"] for upload in uploads]
        last_local_model_paths = [upload["last_local_model_path"] for upload in uploads]
        return models, accs, last_local_model_paths

    def save(
        self,
        *args,
        **kwargs
    ):
        # """
        #     Save model,meta-info,states
        # """
        # eval_log_path1 = f"./log/full_save/server/round{self.elapsed_comm_rounds}_model.pickle"
        # eval_log_path2 = f"./log/full_save/server/round{self.elapsed_comm_rounds}_dict.pickle"
        # if self.args.report_verbosity:
        #     log_obj(eval_log_path1, self.model)
        pass

    def upload(
        self,
        clients,
        *args,
        **kwargs
    ) -> None:
        """
            Upload global model to clients
        """
        for client in clients:
            # make pruning permanent and then upload the model to clients
            model_copy = copy_model(self.model, self.args.device)
            init_model_copy = copy_model(self.init_model, self.args.device)

            params = get_prune_params(model_copy, name='weight')
            for param, name in params:
                prune.remove(param, name)

            init_params = get_prune_params(init_model_copy)
            for param, name in init_params:
                prune.remove(param, name)
            # call client method
            client.download(model_copy, init_model_copy)
