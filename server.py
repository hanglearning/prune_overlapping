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

from sklearn.cluster import KMeans

class Server():
    """
        Central Server
    """

    def __init__(
        self,
        args,
        model,
        init_global_model_path,
        clients,
        global_test_loader
    ):
        super().__init__()
        self.clients = clients
        self.num_clients = len(self.clients)
        self.args = args
        self.model = model
        self.init_model = copy_model(model, self.args.device)
        self.last_global_model_path = init_global_model_path

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

    def model_validation(self, idx_to_last_local_model_path):
        
        """
        Returns:
            list: a list of client idxes identified as legitimate clients
        """
        # if self.args.CELL:
        #     converging_ratio = 1 - self.args.prune_threshold
        # if self.args.overlapping_prune:
        #     converging_ratio = (1 - self.args.prune_threshold)/self.args.overlapping_threshold

        # get layers
        with open(self.last_global_model_path, 'rb') as f:
            layer_to_weights = pickle.load(f)
        layers = list(layer_to_weights.keys())
        num_layers = len(layers)

        # 2 groups of models and treat the majority group as legitimate (following the legitimate direction)
        layer_to_ratios = {l:[] for l in layers} # in the order of client
        client_to_points = {}
        # client_to_layer_to_ratios = {c: {l: [] for l in layers} for c in idx_to_last_local_model_path.keys()}
        for client_idx, local_model_path in idx_to_last_local_model_path.items():
            layer_to_mask = calculate_overlapping_mask([self.last_global_model_path, local_model_path], self.args.check_whole, self.args.overlapping_threshold)
            for layer, mask in layer_to_mask.items():
                # overlapping_ratio = round((mask == 1).sum()/mask.size, 3)
                overlapping_ratio = (mask == 1).sum()/mask.size
                layer_to_ratios[layer].append(overlapping_ratio)
                # client_to_layer_to_ratios[client_idx][layer] = overlapping_ratio
            client_to_points[client_idx] = 0

        # group clients based on ratio
        kmeans = KMeans(n_clusters=2, random_state=0) 
        for layer, ratios in layer_to_ratios.items():
            group_0 = []
            group_1 = []
            kmeans.fit(np.array(ratios).reshape(-1,1))
            for client_iter in range(1, len(kmeans.labels_) + 1):
                label = kmeans.labels_[client_iter - 1]
                if label == 0:
                    group_0.append(client_iter)
                else:
                    group_1.append(client_iter)
            
            if len(group_0) > len(group_1):
                benigh_group = group_0
            elif len(group_0) < len(group_1):
                benigh_group = group_1
            else:
                # when num of two groups are the same, pick the group having higher total overlapping ratios
                group_0_overall_ratios = sum([ratios[client_idx - 1] for client_idx in group_0])
                group_1_overall_ratios = sum([ratios[client_idx - 1] for client_idx in group_1])

                benigh_group = group_1 if group_1_overall_ratios > group_0_overall_ratios else group_0 # very rare when they equal 

            for client_iter in benigh_group:
                client_to_points[client_iter] += 1
        
        return [client_idx for client_idx in client_to_points if client_to_points[client_idx] > num_layers * 0.5] # was >=

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

        global_prune_perc = get_prune_summary(model=self.model,
                                       name='weight')['global']
        print('Global model prune percentage before starting: {}'.format(global_prune_perc))
        wandb.log({f"global_model_pruned_percentage_at_the_beginning": global_prune_perc, "comm_round": comm_round})

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
        idx_to_model, idx_to_acc, idx_to_last_local_model_path = self.download(clients)

        # validation
        identified_benigh_clients = [self.clients[i].idx for i in client_idxs]
        if self.args.validate:
            identified_benigh_clients = self.model_validation(idx_to_last_local_model_path)
        benigh_clients = [self.clients[i].idx for i in client_idxs if not self.clients[i].is_malicious]
        # evaluate validation
        false_positive = 0
        for benigh_client in identified_benigh_clients:
            if self.clients[benigh_client - 1].is_malicious:
                false_positive += 1
        try:
            correct_rate = 1 - false_positive/self.args.n_malicious
        except ZeroDivisionError:
            correct_rate = 1
        print(f"Identified benigh clients {identified_benigh_clients}, total {len(identified_benigh_clients)}.")
        print(f"{false_positive} in {self.args.n_malicious} identified wrong. Correct rate - {correct_rate:.2%}")
        wandb.log({f"correct_rate": correct_rate, "comm_round": comm_round})

        benigh_models = [idx_to_model[c] for c in identified_benigh_clients]
        benigh_model_paths = [idx_to_last_local_model_path[c] for c in identified_benigh_clients]

        # log acc
        if self.args.CELL or self.args.overlapping_prune:
            model_type = "ticket"
        if self.args.fedavg_no_prune:
            model_type = "local"

        iden_benigh_acc = [idx_to_acc[c] for c in identified_benigh_clients]
        benigh_acc = [idx_to_acc[c] for c in benigh_clients]
        iden_benigh_avg_accuracy = np.mean(iden_benigh_acc, axis=0, dtype=np.float32)
        benigh_avg_accuracy = np.mean(benigh_acc, axis=0, dtype=np.float32)
        all_avg_accuracy = np.mean(list(idx_to_acc.values()), axis=0, dtype=np.float32)
        print('-----------------------------', flush=True)
        print(f'| Benigh Average {model_type.title()} Model Accuracy on Local Test Sets: {benigh_avg_accuracy}  | ', flush=True)
        print('-----------------------------', flush=True)
        wandb.log({f"iden_benigh_avg_{model_type}_model_local_acc": iden_benigh_avg_accuracy, "comm_round": comm_round})
        wandb.log({f"benigh_avg_{model_type}_model_local_acc": benigh_avg_accuracy, "comm_round": comm_round})
        wandb.log({f"all_avg_{model_type}_model_local_acc": all_avg_accuracy, "comm_round": comm_round})


        # compute average-model
        aggr_model = self.aggr(benigh_models, clients)

        # test UNpruned global model on entire test set
        if False:
            global_test_acc = util_test(aggr_model,
                                self.global_test_loader,
                                self.args.device,
                                self.args.test_verbose)['Accuracy'][0]
            print('-----------------------------', flush=True)
            print(f'| Un-pruned Global Model on Global Test Set Accuracy at Round {comm_round} : {global_test_acc}  | ', flush=True)
            print('-----------------------------', flush=True)
            wandb.log({"unpruned_global_model_global_acc": global_test_acc, "comm_round": comm_round})

         # test UNpruned global model on each local test set
        if False:
            for client in self.clients:
                global_model_local_set_acc = client.eval(aggr_model)["Accuracy"][0]
                print(f"Un-pruned Global Model on Client {client.idx} Local Test Set Accuracy at Round {comm_round} : {global_model_local_set_acc}")
                wandb.log({f"{client.idx}_unpruned_global_model_local_acc": global_model_local_set_acc, "comm_round": comm_round})

        layer_TO_if_pruned = [False]
        if self.args.overlapping_prune:
            layer_TO_if_pruned, layer_TO_pruned_percentage = prune_by_top_overlap_l1(aggr_model, benigh_model_paths, self.args.check_whole, self.args.overlapping_threshold, self.args.prune_threshold)
            # log pruned amount of each layer
            for layer, pruned_percentage in layer_TO_pruned_percentage.items():
                print(f"Pruned percentage of {layer}: {pruned_percentage:.2%}")
                wandb.log({f"{layer}_pruned_percentage": pruned_percentage, "comm_round": comm_round})

        # if not self.args.overlapping_prune:
        # copy aggregated-model's params to self.model (keep buffer same)
        # source_params = dict(aggr_model.named_parameters())
        # for name, param in self.model.named_parameters():
        #     param.data.copy_(source_params[name])
        self.model = aggr_model

        # save global model
        if self.args.save_global_models:
            model_save_path = f"{self.args.log_dir}/models_weights/globals_0"
            trainable_model_weights = get_trainable_model_weights(self.model)
            self.last_global_model_path = f"{model_save_path}/R{comm_round}.pkl"
            with open(self.last_global_model_path, 'wb') as f:
                pickle.dump(trainable_model_weights, f)

        # test PRUNED global model on entire test set
        if False:
            global_test_acc = util_test(self.model,
                                self.global_test_loader,
                                self.args.device,
                                self.args.test_verbose)['Accuracy'][0]
            print('-----------------------------', flush=True)
            print(f'| Pruned Global Model on Global Test Set Accuracy at Round {comm_round} : {global_test_acc}  | ', flush=True)
            print('-----------------------------', flush=True)
            wandb.log({"pruned_global_model_global_acc": global_test_acc, "comm_round": comm_round})
        if False:
            # test PRUNED global model on each local test set
            for client in self.clients:
                global_model_local_set_acc = client.eval(self.model)["Accuracy"][0]
                print(f"Pruned Global Model on Client {client.idx} Local Test Set Accuracy at Round {comm_round} : {global_model_local_set_acc}")
                wandb.log({f"{client.idx}_pruned_global_model_local_acc": global_model_local_set_acc, "comm_round": comm_round})

        # log global model prune percentage
        if self.args.CELL:
            global_global_prune_perc = get_prune_summary(model=self.model,
                                        name='weight')['global']
            print(f'Global model prune percentage at the end: {global_global_prune_perc}')
            wandb.log({f"global_model_prune_percentage": global_global_prune_perc, "comm_round": comm_round})

        if self.args.overlapping_prune and True in layer_TO_if_pruned.values():
            # reinit - check mask
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
        idx_to_acc = {upload["idx"]: upload["acc"] for upload in uploads}
        idx_to_model = {upload["idx"]: upload["model"] for upload in uploads}
        idx_to_last_local_model_path = {upload["idx"]: upload["last_local_model_path"] for upload in uploads}

        return idx_to_model, idx_to_acc, idx_to_last_local_model_path

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
