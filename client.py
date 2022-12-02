import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import Module
import numpy as np
import os
from typing import Dict
import copy
import math
import wandb
from torch.nn.utils import prune
from util import get_prune_summary, l1_prune, get_prune_params, copy_model
from util import train as util_train
from util import test as util_test
from util import *
from pathlib import Path
from util_dev import *

class Client():
    def __init__(
        self,
        idx,
        args,
        is_malicious,
        train_loader=None,
        test_loader=None,
        user_labels = None,
        global_test_loader=None,
        **kwargs
    ):
        self.idx = idx
        self.args = args
        self.is_malicious = is_malicious
        self.test_loader = test_loader
        self.train_loader = train_loader
        self.user_labels = user_labels
        self.global_test_loader = global_test_loader

        self.eita_hat = self.args.eita
        self.eita = self.eita_hat
        self.alpha = self.args.alpha
        self.num_data = len(self.train_loader)

        # self.elapsed_comm_rounds = 0

        self.accuracies = []
        self.losses = []
        self.prune_rates = []
        self.cur_prune_rate = 0.00

        self.model = None
        self.global_model = None
        self.global_init_model = None

        self.last_local_model_path = None

    def poison_model(self, comm_round, percent=0.2):

        """ Introduce Gaussian Noise to the model weights
            percent: how much on the TOP positions to introduce noise
                    special case: when percent = 1, meaning whole network attack, the noise will ignore the mask sign. see "if percent != 1:"
        """

        if percent == 1:
            # poisoning the whole network, the client introduces signed noise independent of mask sign.
            for layer, module in self.model.named_children():
                for name, weight_params in module.named_parameters():
                    if "weight" in name:
                        noise = self.args.noise_variance * torch.randn(weight_params.size())
                        weight_params.add_(noise.to(self.args.device))
            print(f"Client {self.idx} poisoned the whole network with variance {self.args.noise_variance}.")
        else:
            layer_to_top_mask = generate_2d_magnitude_mask("top", self.last_local_model_path, percent, keep_sign=True)
            layer_to_low_mask = generate_2d_magnitude_mask("low", self.last_local_model_path, 1 - percent, keep_sign=True)
            # the client performs targeted noise attack to keep the sign of magnitude
            for layer, module in self.model.named_children():
                for name, weight_params in module.named_parameters():
                    if "weight" in name:
                        noise = self.args.noise_variance * torch.randn(weight_params.size())
                        ''' increase magnitude for top positions '''
                        top_mask = layer_to_top_mask[layer + "." + name]
                        signed_noise = abs(noise) * top_mask
                        weight_params.add_(signed_noise.to(self.args.device))
                        # sanity check
                        # print((np.array(abs(weight_params) > abs(ori_weight_params)).astype(int) == abs(top_mask)).sum() == top_mask.size)
                        ''' decrease magnitude for low positions '''
                        low_mask = layer_to_low_mask[layer + "." + name]
                        # need special dealing with noise to avoid adding magnitude
                        special_noise = np.minimum(abs(noise), abs(weight_params)) * low_mask * -1
                        ori_weight_params = copy.deepcopy(weight_params)
                        weight_params.add_(special_noise.to(self.args.device))
                        # sanity check
                        # print((np.array(abs(weight_params) < abs(ori_weight_params)).astype(int) == abs(low_mask)).sum() == low_mask.size)

            print(f"Client {self.idx} poisoned top {percent:.2%} and low {1 - percent:.2%} positions with variance {self.args.noise_variance}.")

    def save_model_weights_to_log(self, comm_round, epoch):
        L_or_M = "M" if self.is_malicious else "L"
        model_save_path = f"{self.args.log_dir}/models_weights/{L_or_M}_{self.user_labels}_{self.idx}"
        Path(model_save_path).mkdir(parents=True, exist_ok=True)
        trainable_model_weights = get_trainable_model_weights(self.model)
        # save the current (last) local model weights path
        self.last_local_model_path = f"{model_save_path}/R{comm_round}_E{epoch}.pkl"
        with open(self.last_local_model_path, 'wb') as f:
            pickle.dump(trainable_model_weights, f)


    def update_standalone(self) -> None:
        
        print(f"\nClient {self.idx} doing standalone learning")
        self.model = self.global_model
        
        for comm_round in range(1, self.args.rounds + 1):
            
            print("\nRound", comm_round)
            
            self.train()
            metrics = self.eval(self.model)
            acc = metrics["Accuracy"][0]
            print(f'Trained model accuracy: {acc}')

            wandb.log({f"{self.idx}_acc": acc, "comm_round": comm_round})

    def update_fedavg_no_prune(self, comm_round):
        self.model = self.global_model

        print(f"\n----Client:{self.idx} ({self.user_labels}) FedAvg without Pruning Update----")

        metrics = self.eval(self.global_model)
        accuracy = metrics['Accuracy'][0]
        print(f'Global model local accuracy before training: {accuracy}')

        # train
        self.train(comm_round)
        local_acc = self.eval(self.model)["Accuracy"][0]
        print(f'Trained local model accuracy: {local_acc}')
        wandb.log({f"{self.idx}_local_model_local_acc": local_acc, "comm_round": comm_round})

        if self.is_malicious:
            # poison the last local model
            self.poison_model(comm_round, self.args.noise_targeted_percent)
            poinsoned_acc = self.eval(self.model)["Accuracy"][0]
            print(f'Poisoned accuracy: {poinsoned_acc}, decreased {local_acc - poinsoned_acc}.')
            # overwrite last local model weights
            self.save_model_weights_to_log(comm_round, self.args.epochs)

        # save last local model
        L_or_M = "M" if self.is_malicious else "L"
        model_save_path = f"{self.args.log_dir}/local_models/R_{comm_round}"
        Path(model_save_path).mkdir(parents=True, exist_ok=True)
        torch.save(self.model, f"{model_save_path}/{L_or_M}_{self.user_labels}_{self.idx}.localmodel")

    def update_overlapping_prune(self, comm_round):
        self.model = self.global_model

        produce_mask_from_model(self.model)

        print(f"\n----Client:{self.idx} ({self.user_labels}) Overlapping Pruning Update----")

        metrics = self.eval(self.global_model)
        accuracy = metrics['Accuracy'][0]
        print(f'Global model local accuracy before training: {accuracy}')

        # train
        self.train(comm_round)
        local_acc = self.eval(self.model)["Accuracy"][0]
        print(f'Trained local model accuracy: {local_acc}')
        wandb.log({f"{self.idx}_local_model_local_acc": local_acc, "comm_round": comm_round})

        if self.is_malicious:
            # poison the last local model
            self.poison_model(comm_round, self.args.noise_targeted_percent)
            poinsoned_acc = self.eval(self.model)["Accuracy"][0]
            print(f'Poisoned accuracy: {poinsoned_acc}, decreased {local_acc - poinsoned_acc}.')
            # overwrite last local model weights
            self.save_model_weights_to_log(comm_round, self.args.epochs)

        # save last local model
        L_or_M = "M" if self.is_malicious else "L"
        model_save_path = f"{self.args.log_dir}/local_models/R_{comm_round}"
        Path(model_save_path).mkdir(parents=True, exist_ok=True)
        torch.save(self.model, f"{model_save_path}/{L_or_M}_{self.user_labels}_{self.idx}.localmodel")

    

    #     def prune_trigger(acc_standard):

    #         # add acc_standard to acc_increase_record
    #         if acc_standard >= self.acc_increase_record[-1]:
    #             if len(self.acc_increase_record) > 1:
    #                 if acc_standard - self.acc_increase_record[-1] < self.acc_increase_record[-1] - self.acc_increase_record[-2]:
    #                     # increase slower
    #                     self.acc_increase_record.append(acc_standard)
    #                 else:
    #                     # increase faster or equally
    #                     self.acc_increase_record = [self.acc_increase_record[-1]]
    #             else:
    #                 self.acc_increase_record.append(acc_standard)

                    
    #         if len(self.acc_increase_record) < self.args.beta + 1:
    #             self.prune_trigger = False
    #         judge_list = []
    #         for i in range(1, self.args.beta + 1):
    #             judge_list.append(self.acc_increase_record[i] - self.acc_increase_record[i-1])



            
    #         self.prune_trigger = True
    #         self.acc_increase_record = [self.acc_increase_record[-1]]

    #     print(f"\n----------Client:{self.idx} SPEED Update---------------------")

    #     metrics = self.eval(self.global_model)
    #     global_model_local_acc = metrics['Accuracy'][0]
    #     print(f'Global model local accuracy before pruning and training: {global_model_local_acc}')

    #     # global model prune percentage
    #     prune_rate = get_prune_summary(model=self.global_model, name='weight')['global']

    #     do_prune = False
    #     if prune_rate < self.args.prune_threshold:
    #         # determine if prune
    #         do_prune = prune_trigger(global_model_local_acc)

    #     if do_prune:
    #         self.cur_prune_rate = min(self.cur_prune_rate + self.args.prune_step,
    #                                       self.args.prune_threshold)
    #         l1_prune(model=self.global_model,
    #                     amount=self.cur_prune_rate - prune_rate, #TODO - examine correct prune amount
    #                     name='weight',
    #                     verbose=self.args.prune_verbose)
    #         self.prune_rates.append(self.cur_prune_rate)

    #         # reinitialize model with init_params
    #         source_params = dict(self.global_init_model.named_parameters())
    #         for name, param in self.global_model.named_parameters():
    #             param.data.copy_(source_params[name].data)
            
    #         # reinit prune trigger
    #         self.prune_trigger = False
    #     else:
    #         self.prune_rates.append(prune_rate)
        
    #     # continue to train if not prune
    #     self.model = self.global_model

    #     self.train()

    #     ticket_acc = self.eval(self.model)["Accuracy"][0]
    #     print(f'Trained model accuracy: {ticket_acc}')
    #     wandb.log({f"{self.idx}_ticket_local_acc": ticket_acc, "comm_round": comm_round})

    #     wandb.log({f"{self.idx}_cur_prune_rate": self.cur_prune_rate})
    #     wandb.log(
    #         {f"{self.idx}_percent_pruned": self.prune_rates[-1]}) # model sparsity at this moment



    # original CELL code, do not touch
    def update_CELL(self, comm_round) -> None:
        """
            Interface to Server
        """
        print(f"\n----------Client:{self.idx} CELL Update---------------------")

        metrics = self.eval(self.global_model)
        accuracy = metrics['Accuracy'][0]
        print(f'Global model local accuracy before pruning and training: {accuracy}')

        # global model prune percentage
        prune_rate = get_prune_summary(model=self.global_model, name='weight')['global']
           
        if self.cur_prune_rate < self.args.prune_threshold:
            if accuracy > self.eita:
                self.cur_prune_rate = min(self.cur_prune_rate + self.args.prune_step,
                                          self.args.prune_threshold)
                if self.cur_prune_rate > prune_rate:
                    l1_prune(model=self.global_model,
                             amount=self.cur_prune_rate - prune_rate,
                             name='weight',
                             verbose=self.args.prune_verbose)
                    self.prune_rates.append(self.cur_prune_rate)
                else:
                    self.prune_rates.append(prune_rate)
                # reinitialize model with init_params
                source_params = dict(self.global_init_model.named_parameters())
                for name, param in self.global_model.named_parameters():
                    param.data.copy_(source_params[name].data)

                self.model = self.global_model
                self.eita = self.eita_hat

            else:
                self.eita *= self.alpha
                self.model = self.global_model
                self.prune_rates.append(prune_rate)
        else:
            if self.cur_prune_rate > prune_rate:
                l1_prune(model=self.global_model,
                         amount=self.cur_prune_rate-prune_rate,
                         name='weight',
                         verbose=self.args.prune_verbose)
                self.prune_rates.append(self.cur_prune_rate)
            else:
                self.prune_rates.append(prune_rate)
            self.model = self.global_model

        print(f"\nTraining local model")
        self.train()

        ticket_acc = self.eval(self.model)["Accuracy"][0]
        print(f'Trained model accuracy: {ticket_acc}')
        wandb.log({f"{self.idx}_ticket_local_acc": ticket_acc, "comm_round": comm_round})

        wandb.log({f"{self.idx}_cur_prune_rate": self.cur_prune_rate}) # client's prune rate, but not necessarily the same as _percent_pruned because when < validation_threshold, no prune and use the whole model
        # wandb.log({f"{self.idx}_eita": self.eita}) - I don't care logging it
        wandb.log(
            {f"{self.idx}_percent_pruned": self.prune_rates[-1]}) # model sparsity at this moment


    def train(self, comm_round):
        """
            Train NN
        """
        losses = []

        for epoch in range(1, self.args.epochs + 1):
            if self.args.train_verbose:
                print(
                    f"Client={self.idx}, epoch={epoch}")

            metrics = util_train(self.model,
                                 self.train_loader,
                                 self.args.optimizer,
                                 self.args.lr,
                                 self.args.device,
                                 self.args.train_verbose)
            losses.append(metrics['Loss'][0])

            # save intermediate model
            self.save_model_weights_to_log(comm_round, epoch)

        self.losses.extend(losses)

    @torch.no_grad()
    def download(self, global_model, global_init_model, *args, **kwargs):
        """
            Download global model from server
        """
        self.global_model = global_model
        self.global_init_model = global_init_model

        params_to_prune = get_prune_params(self.global_model)
        for param, name in params_to_prune:
            weights = getattr(param, name)
            masked = torch.eq(weights.data, 0.00).sum().item()
            # masked = 0.00
            prune.l1_unstructured(param, name, amount=int(masked))

        params_to_prune = get_prune_params(self.global_init_model)
        for param, name in params_to_prune:
            weights = getattr(param, name)
            masked = torch.eq(weights.data, 0.00).sum().item()
            # masked = 0.00
            prune.l1_unstructured(param, name, amount=int(masked))

    def eval(self, model):
        """
            Eval self.model
        """
        eval_score = util_test(model,
                               self.test_loader,
                               self.args.device,
                               self.args.test_verbose)
        self.accuracies.append(eval_score['Accuracy'][0])
        return eval_score

    def save(self, *args, **kwargs):
        pass

    def upload(self, *args, **kwargs) -> Dict[nn.Module, float]:
        """
            Upload self.model
        """
        upload_model = copy_model(model=self.model, device=self.args.device)
        params_pruned = get_prune_params(upload_model, name='weight')
        for param, name in params_pruned:
            prune.remove(param, name)
        return {
            'model': upload_model,
            'last_local_model_path': self.last_local_model_path,
            'acc': self.accuracies[-1]
        }
