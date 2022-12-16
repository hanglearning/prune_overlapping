import pickle
import math
import numpy as np
import torch
import sys
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mtick
import torch.nn.utils.prune as prune

from util import *
from util import test as util_test

import ast
import random
import os
from os import listdir
from os.path import isfile, join


def get_trainable_model_weights(model):
    """
    Args:
        model (_torch model_): NN Model

    Returns:
        layer_to_param _dict_: you know!
    """
    layer_to_param = {} 
    for layer_name, param in model.named_parameters():
        if 'weight' in layer_name:
            layer_to_param[layer_name.split('.')[0]] = param.cpu().detach().numpy()
    return layer_to_param

def check_sparsity_based_on_mask(model):
    for layer, module in model.named_children():
        for name, mask in module.named_buffers():
            if 'mask' in name:
                print(f"{layer} sparsity - {(mask == 1).sum()/torch.numel(mask)}")

def generate_2d_top_magnitude_mask(model_path, percent, check_whole = False, keep_sign = False, pruning_two = True):

    """
        returns 2d top magnitude mask.
        1. keep_sign == True
            it keeps the sign of the original weight. Used in introduce noise. 
            returns mask with -1, 1, 0.
        2. keep_sign == False
            calculate absolute magitude mask. Used in calculating weight overlapping.
            returns binary mask with 1, 0.
        
        *. pruning_two: this is useful for the function calculate_top_overlapping_ratio(). When pruning_two = True, it returns 1 as top percent, 2 as pruned positions and 0 as otherwise. This is used in calculate_top_overlapping_ratio() to consider pruned weights when we calculate the top_overlapping_ratio.
        
        For example, if the validation area is top 50% and the network reaches its target sparsity 20%, if we set pruning_two = False, calculate_top_overlapping_ratio() returns 40%. If we set pruning_two = True, calculate_top_overlapping_ratio() returns 100%. 
    """
    
    layer_to_mask = {}

    with open(model_path, 'rb') as f:
        nn_layer_to_weights = pickle.load(f)
            
    for layer, param in nn_layer_to_weights.items():
    
        # take abs as we show magnitude values
        abs_param = np.absolute(param)

        mask_2d = np.empty_like(abs_param)
        mask_2d[:] = 0 # initialize as 0

        base_size = abs_param.size if check_whole else abs_param.size - abs_param[abs_param == 0].size

        top_boundary = math.ceil(base_size * percent)
                    
        percent_threshold = -np.sort(-abs_param.flatten())[top_boundary]

        # change top weights to 1
        mask_2d[np.where(abs_param > percent_threshold)] = 1

        # sanity check
        # one_counts = (mask_2d == 1).sum()
        # print(one_counts/param.size)

        layer_to_mask[layer] = mask_2d
        if keep_sign:
            layer_to_mask[layer] *= np.sign(param)

    # sanity check
    # for layer in layer_to_mask:
	#     print((layer_to_mask[layer] == 1).sum()/layer_to_mask[layer].size)

    return layer_to_mask

def calculate_overlapping_mask(model_paths, check_whole, percent):
    layer_to_masks = []

    for model_path in model_paths:
        layer_to_masks.append(generate_2d_top_magnitude_mask(model_path, percent, check_whole))

    ref_layer_to_mask = layer_to_masks[0]

    for layer_to_mask_iter in range(len(layer_to_masks[1:])):
        layer_to_mask = layer_to_masks[1:][layer_to_mask_iter]
        for layer, mask in layer_to_mask.items():
            ref_layer_to_mask[layer] *= mask
            if check_whole:
                # for debug - when each local model has high overlapping with the last global model, why the overlapping ratio for all local models seems to be low?
                print(f"iter {layer_to_mask_iter + 1}, layer {layer} - overlapping ratio on top {percent:.2%} is {(ref_layer_to_mask[layer] == 1).sum()/ref_layer_to_mask[layer].size/percent:.2%}")
        print()

    return ref_layer_to_mask


def calculate_top_overlapping_ratio(model_paths, percent, check_whole):
    """
    Args:
        model_paths - list of model paths
    Note - the ratio is based on the top percent area, rather than whole network.
    """
    layer_to_masks = []

    for model_path in model_paths:
        # layer_to_masks.append(generate_1d_top_magnitude_binary_mask(model_path, percent))
        layer_to_masks.append(generate_2d_top_magnitude_mask(model_path, percent, check_whole))

    ref_layer_to_mask = layer_to_masks[0]

    for layer_to_mask in layer_to_masks[1:]:
        for layer, mask in layer_to_mask.items():
            ref_layer_to_mask[layer] *= mask
    
    layer_to_overlapping_ratio = {}
    for layer, mask in ref_layer_to_mask.items():
        one_counts = (mask == 1).sum()
        ratio = one_counts/(mask.size * percent)
        # print(f"{layer} overlapping ratio: {ratio:.2%}")
        layer_to_overlapping_ratio[layer] = ratio

    return layer_to_overlapping_ratio

def get_pruned_percent_by_weights(model):
    total_params_count = get_num_total_model_params(model)
    total_0_count = 0
    total_nan_count = 0
    layer_to_pruned_amount = {}
    for layer, module in model.named_children():
        for name, weight_params in module.named_parameters():
            if 'weight' in name:
                if weight_params.is_cuda:
                    total_0_count += len(list(zip(*np.where(weight_params.cpu() == 0))))
                    total_nan_count += len(torch.nonzero(torch.isnan(weight_params.cpu().view(-1))))
                else:
                    total_0_count += len(list(zip(*np.where(weight_params == 0))))
                    total_nan_count += len(torch.nonzero(torch.isnan(weight_params.view(-1))))
    if total_nan_count > 0:
        sys.exit("nan bug")
    return total_0_count / total_params_count

def get_pruned_percent_by_layer_weights(model):
    layer_TO_pruned_amount = {}
    for layer, module in model.named_children():
        for name, weight_params in module.named_parameters():
            if 'weight' in name:
                if weight_params.is_cuda:
                    total_0_count = len(list(zip(*np.where(weight_params.cpu() == 0))))
                else:
                    total_0_count = len(list(zip(*np.where(weight_params == 0))))
                layer_TO_pruned_amount[layer] = total_0_count/weight_params.numel()
    return layer_TO_pruned_amount


def get_pruned_percent_by_mask(model):
    total_params_count = get_num_total_model_params(model)
    total_0_count = 0
    for layer, module in model.named_children():
        for name, mask in module.named_buffers():
            if 'mask' in name:
                if mask.is_cuda:
                    total_0_count += len(list(zip(*np.where(mask.cpu() == 0))))
                else:
                    total_0_count += len(list(zip(*np.where(mask == 0))))
    return total_0_count / total_params_count

def get_pruned_percent_by_layer_mask(model):
    layer_TO_pruned_percent = {}
    for layer, module in model.named_children():
        for name, mask in module.named_buffers():
            if 'mask' in name:
                layer_TO_pruned_percent[layer] = (mask == 0).sum()/torch.numel(mask)
    return layer_TO_pruned_percent


def get_num_total_model_params(model):
    total_num_model_params = 0
    # not including bias
    for layer_name, params in model.named_parameters():
        if 'weight' in layer_name:
            total_num_model_params += params.numel()
    return total_num_model_params

def global_vs_local_overlapping_by_round(log_folder, comm_round, percent, check_whole=False):
    """Plot overlapping ratios change through epochs between global model and intermediate local models
        # rep_device - used as the baseline to determine common labels color
    """
    global_model_path = f"{log_folder}/models_weights/globals_0/R{comm_round - 1}.pkl"
    # get layers
    with open(global_model_path, 'rb') as f:
        ref_nn_layer_to_weights = pickle.load(f)
    layers = list(ref_nn_layer_to_weights.keys())
    
    clients = [name for name in os.listdir(f"{log_folder}/models_weights/") if os.path.isdir(os.path.join(f"{log_folder}/models_weights/", name))]
    clients.sort(key=lambda x: int(x.split('_')[-1]))
    clients.remove("globals_0")

    layer_to_clients_to_overlappings = {l: {c: [] for c in clients} for l in layers}

    client_to_xticks = {}
    for client in clients:
        models_folder = f"{log_folder}/models_weights/{client}"
        model_files_this_round = [f for f in os.listdir(models_folder) if os.path.isfile(os.path.join(models_folder, f)) and f"R{comm_round}_" in f]
        model_files_this_round.sort(key=lambda x: int(x.split('E')[1].split(".")[0]))
        client_to_xticks[client] = [f.split(".")[0] for f in model_files_this_round]
        
        for epoch_file in model_files_this_round:
            epoch_file_path = f"{log_folder}/models_weights/{client}/{epoch_file}"
            layer_to_ratio = calculate_top_overlapping_ratio([global_model_path, epoch_file_path], percent, check_whole)
            for layer, ratio in layer_to_ratio.items():
                layer_to_clients_to_overlappings[layer][client].append(ratio)

    for layer, clients_to_overlappings in layer_to_clients_to_overlappings.items():
        plt.figure(dpi=250)
        for client, overlappings in clients_to_overlappings.items():
            plot_y = [percentage * 100 for percentage in overlappings] # used to format percentage
            # plt.plot(client_to_xticks[client], plot_y, color=colors[iter - 1])
            plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(decimals=2))
            if client[0] == 'L':
                plt.plot(client_to_xticks[client], plot_y, color='green')
            else:
                plt.plot(client_to_xticks[client], plot_y, color='red')
            # plt.annotate(client, xy=(iter % len(client_to_xticks[client]), plot_y[iter % len(plot_y)]), size=8, color=colors[iter - 1])
            # plt.annotate(client, xy=(len(overlappings) - iter * 0.8, overlappings[iter % len(overlappings)]), size=6, color=colors[iter - 1])

        patch_L = mpatches.Patch(color='green', label=f'Legitimate')
        patch_M = mpatches.Patch(color='red', label=f'Malicious')

        plt.legend(handles=[patch_L, patch_M], loc='best')

        plt.xlabel('Epoch')
        plt.ylabel('Overlapping Ratio')
        plt.title(f"Global R{comm_round - 1} vs. Local R{comm_round} - {layer}")

        plot_save_folder = f"{log_folder}/plots/global_vs_local/R{comm_round}"
        os.makedirs(plot_save_folder, exist_ok=True)

        plt.savefig(f"{plot_save_folder}/global_vs_local_{layer}.png")


# for i in range(1, 31):
#     global_vs_local_overlapping_by_round("/Users/chenhang/Downloads/CELL/12142022_213858_SEED_40_NOISE_PERCENT_1.0_VAR_1.0", i, 0.2, True)

def global_vs_local_overlapping_all_rounds(log_folder, percent, check_whole=False):
    """Plot overlapping ratios change through epochs between global model and intermediate local models
        # rep_device - used as the baseline to determine common labels color
    """
    total_rounds = len([f for f in os.listdir(f"{log_folder}/models_weights/globals_0")])
    init_model_path = f"{log_folder}/models_weights/globals_0/R0.pkl"
    # get layers
    with open(init_model_path, 'rb') as f:
        ref_nn_layer_to_weights = pickle.load(f)
    layers = list(ref_nn_layer_to_weights.keys())

    # get clients
    clients = [name for name in os.listdir(f"{log_folder}/models_weights/") if os.path.isdir(os.path.join(f"{log_folder}/models_weights/", name))]
    clients.sort(key=lambda x: int(x.split('_')[-1]))
    clients.remove("globals_0")

    layer_to_clients_to_overlappings = {k: {c: [] for c in clients} for k in layers}
    client_to_xticks = {c: [] for c in clients}
    
    for comm_round in range(1, total_rounds + 1):
        global_model_path = f"{log_folder}/models_weights/globals_0/R{comm_round - 1}.pkl"
        
        for client in clients:
            models_folder = f"{log_folder}/models_weights/{client}"
            model_files_this_round = [f for f in os.listdir(models_folder) if os.path.isfile(os.path.join(models_folder, f)) and f"R{comm_round}_" in f]
            model_files_this_round.sort(key=lambda x: int(x.split('E')[1].split(".")[0]))
            client_to_xticks[client].extend([f.split(".")[0] for f in model_files_this_round])
            
            for epoch_file in model_files_this_round:
                epoch_file_path = f"{log_folder}/models_weights/{client}/{epoch_file}"
                layer_to_ratio = calculate_top_overlapping_ratio([global_model_path, epoch_file_path], percent, check_whole)
                for layer, ratio in layer_to_ratio.items():
                    layer_to_clients_to_overlappings[layer][client].append(ratio)

    plt.figure(dpi=250) # TODO
    for layer, clients_to_overlappings in layer_to_clients_to_overlappings.items():
        for client, overlappings in clients_to_overlappings.items():
            plot_y = [percentage * 100 for percentage in overlappings] # used to format percentage
            # plt.plot(client_to_xticks[client], plot_y, color=colors[iter - 1])
            plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(decimals=2))
            if client[0] == 'L':
                plt.plot(client_to_xticks[client], plot_y, color='green')
            else:
                plt.plot(client_to_xticks[client], plot_y, color='red')
            # plt.annotate(client, xy=(iter % len(client_to_xticks[client]), plot_y[iter % len(plot_y)]), size=8, color=colors[iter - 1])
            # plt.annotate(client, xy=(len(overlappings) - iter * 0.8, overlappings[iter % len(overlappings)]), size=6, color=colors[iter - 1])

        patch_L = mpatches.Patch(color='green', label=f'Legitimate')
        patch_M = mpatches.Patch(color='red', label=f'Malicious')

        plt.legend(handles=[patch_L, patch_M], loc='best')

        plt.xlabel('Epoch')
        plt.ylabel('Overlapping Ratio')
        plt.title(f"Global R{comm_round - 1} vs. Local R{comm_round} - {layer}")

        plot_save_folder = f"{log_folder}/plots/global_vs_local"
        os.makedirs(plot_save_folder, exist_ok=True)

        plt.savefig(f"{plot_save_folder}/global_vs_local_{layer}.png")


def last_local_vs_locals_overlapping(log_folder, ref_client, comm_round, last_epoch, percent, color_M=True, check_whole=False):
    """Plot overlapping ratios change through epochs between a client's last local model and other clients' intermediate and last local models.
       See 
        1. if common labels have more overlapping
        2. if more epochs contribute to more overlapping
        2. if malicious models have low overlapping
        ref_client will be the validator in blockchained app
    """
    ref_local_model_path = f"{log_folder}/models_weights/{ref_client}/R{comm_round}_E{last_epoch}.pkl"
    # get layers
    with open(ref_local_model_path, 'rb') as f:
        ref_nn_layer_to_weights = pickle.load(f)
    layers = list(ref_nn_layer_to_weights.keys())
    
    clients = [name for name in os.listdir(f"{log_folder}/models_weights/") if os.path.isdir(os.path.join(f"{log_folder}/models_weights/", name))]
    clients.sort(key=lambda x: int(x.split('_')[-1]))
    clients.remove("globals_0")
    clients.remove(ref_client)

    layer_to_clients_to_overlappings = {k: {c: [] for c in clients} for k in layers}

    client_to_xticks = {}
    for client in clients:
        models_folder = f"{log_folder}/models_weights/{client}"
        model_files_this_round = [f for f in os.listdir(models_folder) if os.path.isfile(os.path.join(models_folder, f)) and f"R{comm_round}" in f]
        model_files_this_round.sort(key=lambda x: int(x.split('E')[1].split(".")[0]))
        client_to_xticks[client] = [f.split(".")[0] for f in model_files_this_round]
        
        for epoch_file in model_files_this_round:
            epoch_file_path = f"{log_folder}/models_weights/{client}/{epoch_file}"
            layer_to_ratio = calculate_top_overlapping_ratio([ref_local_model_path, epoch_file_path], percent, check_whole)
            for layer, ratio in layer_to_ratio.items():
                layer_to_clients_to_overlappings[layer][client].append(ratio)

    # plot by layer, color by amount of common labels
    matching_colors = {3: "green", 2: "orange", 1: "blue", 0: "grey", "M": "red"}
    ref_client_labels = ast.literal_eval(ref_client.split("_")[1])

    for layer, clients_to_overlappings in layer_to_clients_to_overlappings.items():
        plt.figure(dpi=250)
        iter = 1
        for client, overlappings in clients_to_overlappings.items():
            plot_y = [percentage * 100 for percentage in overlappings] # used to format percentage
            client_labels = ast.literal_eval(client.split("_")[1])
            num_matching_labels = len(set(ref_client_labels) & set(client_labels))
            if client[0] == "M" and color_M:
                num_matching_labels = "M"
            plt.plot(client_to_xticks[client], plot_y, color=matching_colors[num_matching_labels])
            plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(decimals=2))
            plt.annotate(client, xy=(iter % len(client_to_xticks[client]), plot_y[iter % len(plot_y)]), size=8, color=matching_colors[num_matching_labels])
            iter += 1

        patch_3 = mpatches.Patch(color=matching_colors[3], label=f'3 matches')
        patch_2 = mpatches.Patch(color=matching_colors[2], label=f'2 matches')
        patch_1 = mpatches.Patch(color=matching_colors[1], label=f'1 match')
        patch_0 = mpatches.Patch(color=matching_colors[0], label=f'0 match')
        patch_M = mpatches.Patch(color=matching_colors['M'], label=f'Malicious')

        plt.legend(handles=[patch_3, patch_2, patch_1, patch_0, patch_M], loc='best')

        plt.xlabel('Epoch')
        plt.ylabel('Overlapping Ratio')
        plt.title(f"Client {ref_client} vs. Locals in R{comm_round} - {layer}")

        plot_save_folder = f"{log_folder}/plots/R{comm_round}/locals"
        os.makedirs(plot_save_folder, exist_ok=True)

        plt.savefig(f"{plot_save_folder}/{ref_client}_vs_locals_{layer}.png")

# last_local_vs_locals_overlapping("/Users/chenhang/Documents/Temp/11262022_011637_SEED_60_NOISE_1.0", "L_[4, 6, 8]_1", 1, 10, percent, color_M=False)


def test_local_model_acc(validator_ref_client, log_folder, comm_round, color_M=True):
    ref_data_loader_path = f"{log_folder}/data_loaders/{validator_ref_client}.dataloader"
    validation_set = torch.load(ref_data_loader_path)
    
    clients = [name for name in os.listdir(f"{log_folder}/models_weights/") if os.path.isdir(os.path.join(f"{log_folder}/models_weights/", name))]
    clients.sort(key=lambda x: int(x.split('_')[-1]))
    clients.remove("globals_0")
    clients.remove(validator_ref_client)

    # plot by layer, color by amount of common labels
    matching_colors = {3: "green", 2: "orange", 1: "blue", 0: "grey", "M": "red"}
    ref_client_labels = ast.literal_eval(validator_ref_client.split("_")[1])

    clients_acc = []
    bar_colors = []
    for client in clients:
        last_local_model = torch.load(f"{log_folder}/local_models/R_{comm_round}/{client}.localmodel")
        clients_acc.append(util_test(last_local_model, validation_set, device='cpu',)['Accuracy'][0])
        
        client_labels = ast.literal_eval(client.split("_")[1])
        num_matching_labels = len(set(ref_client_labels) & set(client_labels))
        if client[0] == "M" and color_M:
            num_matching_labels = "M"
        bar_colors.append(matching_colors[num_matching_labels])
    
    plt.bar(clients, clients_acc, color = bar_colors)

    patch_3 = mpatches.Patch(color=matching_colors[3], label=f'3 matches')
    patch_2 = mpatches.Patch(color=matching_colors[2], label=f'2 matches')
    patch_1 = mpatches.Patch(color=matching_colors[1], label=f'1 match')
    patch_0 = mpatches.Patch(color=matching_colors[0], label=f'0 match')
    patch_M = mpatches.Patch(color=matching_colors['M'], label=f'Malicious')

    plt.legend(handles=[patch_3, patch_2, patch_1, patch_0, patch_M], loc='best')

    plt.xlabel('Client')
    plt.ylabel('Test Accuracy')
    plt.title(f"Local models of Round {comm_round} on {validator_ref_client}'s Training Set")

    plot_save_folder = f"{log_folder}/plots/acc/R{comm_round}"
    os.makedirs(plot_save_folder, exist_ok=True)
    
    fig = plt.gcf()
    fig.set_size_inches(1.5 * len(clients), 10.5)

    fig.savefig(f"{plot_save_folder}/{validator_ref_client}_validation_acc.png")

# test_local_model_acc("L_[4, 6, 8]_1", "/Users/chenhang/Documents/Temp/11282022_211140_SEED_60_NOISE_PERCENT_0.2_VAR_1.0", 1, color_M=True)
def get_prune_params_with_layer_name(model, name='weight') -> List[Tuple[nn.Parameter, str]]:
    params_to_prune = []
    for layer, module in model.named_children():
        for name_, param in module.named_parameters():
            if 'weight' in name_:
                params_to_prune.append((module, layer))
    return params_to_prune

def prune_by_top_overlap_l1(model, last_local_model_paths, check_whole, overlapping_threshold, prune_threshold):

    top_master_mask = calculate_overlapping_mask(last_local_model_paths, check_whole, overlapping_threshold)
    params_to_prune = get_prune_params_with_layer_name(model)
    layer_TO_if_pruned = {}
    layer_TO_pruned_percentage = {}
    for params, layer in params_to_prune:
        layer_top_mask = top_master_mask[layer]
        # calculate overlapping ratio
        overlapping_ratio = round((layer_top_mask == 1).sum()/torch.numel(params.weight), 3)
        # determine if prune
        old_pruned_percent = round(float((params.weight == 0).sum()/torch.numel(params.weight)), 3)
        print(f"old_pruned_percent - {layer}", old_pruned_percent)
        print(f"overlapping_ratio towards the whole layer - {layer}", overlapping_ratio)
        if old_pruned_percent < prune_threshold:
            new_pruned_percent = min(old_pruned_percent + overlapping_ratio, prune_threshold)
            prune.l1_unstructured(params, 'weight', new_pruned_percent)
            layer_TO_if_pruned[layer] = True
            layer_TO_pruned_percentage[layer] = new_pruned_percent
            print(f"{layer} pruned {new_pruned_percent:.2%}")
        else:
            prune.l1_unstructured(params, 'weight', old_pruned_percent)
            layer_TO_if_pruned[layer] = False
            layer_TO_pruned_percentage[layer] = old_pruned_percent
            print(f"{layer} skipeed pruning. Current pruned perc {old_pruned_percent:.2%}")

    return layer_TO_if_pruned, layer_TO_pruned_percentage
                
def produce_mask_from_model(model):
    # use prune with 0 amount to init mask for the model
    # create mask in-place on model
    l1_prune(model=model,
                amount=0.00,
                name='weight',
                verbose=False)
    layer_to_masked_positions = {}
    for layer, module in model.named_children():
        for name, weight_params in module.named_parameters():
            if 'weight' in name:
                if weight_params.is_cuda:
                    layer_to_masked_positions[layer] = list(zip(*np.where(weight_params.cpu() == 0)))
                else:
                    layer_to_masked_positions[layer] = list(zip(*np.where(weight_params == 0)))
        
    for layer, module in model.named_children():
        for name, mask in module.named_buffers():
            if 'mask' in name:
                for pos in layer_to_masked_positions[layer]:
                    mask[pos] = 0

def calc_mask_from_model_without_mask_object(model):
    layer_to_mask = {}
    for layer, module in model.named_children():
        for name, weight_params in module.named_parameters():
            if 'weight' in name:
                layer_to_mask[layer] = np.ones_like(weight_params.cpu())
                layer_to_mask[layer][weight_params.cpu() == 0] = 0
    return layer_to_mask

def calc_mask_from_model_with_mask_object(model):
    layer_to_mask = {}
    for layer, module in model.named_children():
        for name, mask in module.named_buffers():
            if 'mask' in name:
                layer_to_mask[layer] = mask
    return layer_to_mask

class LowOverlappingPrune(prune.BasePruningMethod):
    
    PRUNING_TYPE = 'unstructured'

    def __init__(self, layer_new_mask):
        self.layer_new_mask = layer_new_mask

    def compute_mask(self, t, default_mask):
        return torch.flatten(self.layer_new_mask)

def lowOverlappingPrune(module, layer_new_mask, dev_device, name='weight'):
    layer_new_mask = torch.from_numpy(layer_new_mask).to(dev_device)
    LowOverlappingPrune.apply(module, name, layer_new_mask)
    return module

# class topOverlappingPrune(prune.BasePruningMethod):
    
#     PRUNING_TYPE = 'unstructured'

#     def __init__(self, layer_new_mask):
#         self.layer_new_mask = layer_new_mask

#     def compute_mask(self, t, default_mask):
#         return torch.flatten(self.layer_new_mask)

# def topOverlappingPrune(module, layer_new_mask, dev_device, name='weight'):
#     layer_new_mask = torch.from_numpy(layer_new_mask).to(dev_device)
#     LowOverlappingPrune.apply(module, name, layer_new_mask)
#     return module


def global_vs_last_local_overlapping(log_folder, last_epoch, percent, check_whole=False):
    """Plot overlapping ratios change through epochs between global model and intermediate local models
        # rep_device - used as the baseline to determine common labels color
    """
    # get all available comm rounds
    comm_rounds = [int(f.split('.')[0][1:]) for f in listdir(f"{log_folder}/models_weights/globals_0") if isfile(join(f"{log_folder}/models_weights/globals_0", f))]
    comm_rounds.remove(0)
    comm_rounds.sort()

    clients = [name for name in os.listdir(f"{log_folder}/models_weights/") if os.path.isdir(os.path.join(f"{log_folder}/models_weights/", name))]
    clients.sort(key=lambda x: int(x.split('_')[-1]))
    clients.remove("globals_0")

    ref_global_model = f"{log_folder}/models_weights/globals_0/R0.pkl"
    # get layers
    with open(ref_global_model, 'rb') as f:
        ref_nn_layer_to_weights = pickle.load(f)
    layers = list(ref_nn_layer_to_weights.keys())

    layer_TO_clients = {l:{c: [] for c in clients} for l in layers}

    for comm_round in comm_rounds:

        global_model_path = f"{log_folder}/models_weights/globals_0/R{comm_round - 1}.pkl"
        
        for client in clients:
            local_model_path = f"{log_folder}/models_weights/{client}/R{comm_round}_E{last_epoch}.pkl"
            layer_to_ratio = calculate_top_overlapping_ratio([global_model_path, local_model_path], percent, check_whole) # TODO - buggy when percent = 0.5
            for layer, ratio in layer_to_ratio.items():
                layer_TO_clients[layer][client].append(ratio)

    for layer, client_to_overlappings in layer_TO_clients.items():
        plt.figure(dpi=250)
        for client, overlappings in client_to_overlappings.items():
            plot_y = [percentage * 100 for percentage in overlappings]
            color = "green" if client[0] == "L" else "red"
            plt.plot(comm_rounds, plot_y, color=color)
            plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(decimals=2))
    
        patch_L = mpatches.Patch(color='green', label='Legitimate')
        patch_M = mpatches.Patch(color='red', label=f'Malicious')

        plt.legend(handles=[patch_L, patch_M], loc='best')

        plt.xlabel('Comm Round')
        plt.ylabel('Overlapping Ratio')
        plt.title(f"Global R{comm_round - 1} vs. Local E{last_epoch} - {layer}")

        plot_save_folder = f"{log_folder}/plots/global_local_overlappings/"
        os.makedirs(plot_save_folder, exist_ok=True)

        plt.savefig(f"{plot_save_folder}/global_vs_local_{layer}.png")
    

# global_vs_last_local_overlapping("/Users/chenhang/Documents/Temp/TOP_1.0_12032022_164340_SEED_40_NOISE_PERCENT_1.0_VAR_1.0", last_epoch = 5, percent=0.5)

if __name__ == "__main__":
    # global_vs_local_overlapping_all_rounds("/Users/chenhang/Downloads/CELL/12142022_213858_SEED_40_NOISE_PERCENT_1.0_VAR_1.0", 0.2, True)
    global_vs_local_overlapping_by_round("/Users/chenhang/Downloads/CELL/12142022_213858_SEED_40_NOISE_PERCENT_1.0_VAR_1.0", 28, 0.5, True)