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



def generate_2d_magnitude_mask(top_or_low, model_path, percent=0.2, keep_sign = False, ignore_pruned = True):

    """
        returns 2d mask.
        1. keep_sign == True
            it keeps the sign of the original weight. Used in introduce noise. 
            returns mask with -1, 1, 0.
        2. keep_sign == False
            calculate absolute magitude mask. Used in calculating weight overlapping.
            returns binary mask with 1, 0.
    """
    
    layer_to_mask = {}

    with open(model_path, 'rb') as f:
        nn_layer_to_weights = pickle.load(f)
            
    for layer, param in nn_layer_to_weights.items():
    
        # take abs as we show magnitude values
        abs_param = np.absolute(param)

        pruned_percent = abs_param[abs_param == 0].size/abs_param.size
        
        # if need low, this is a trick
        proxy_percent = percent + pruned_percent if top_or_low == 'low' else percent
            
        percent_order = math.ceil(abs_param.size * proxy_percent)

        abs_param_1d_array = abs_param.flatten()
        if top_or_low == 'top':
            # percent_threshold = np.partition(abs_param_1d_array, -percent_order)[-percent_order]
            percent_threshold = -np.sort(-abs_param_1d_array)[percent_order]
        elif top_or_low == 'low':
            # percent_threshold = np.partition(abs_param_1d_array, percent_order - 1)[percent_order - 1]
            percent_threshold = np.sort(abs_param_1d_array)[percent_order]


        mask_2d = np.empty_like(abs_param)
        mask_2d[:] = 0 # initialize as 0

        if top_or_low == 'top':
            # change top weights to 1
            mask_2d[np.where(abs_param > percent_threshold)] = 1
        elif top_or_low == 'low':
            # change low weights to 1
            if ignore_pruned:
                mask_2d[np.where((0 < abs_param) & (abs_param < percent_threshold))] = 1
            else:
                mask_2d[np.where((0 <= abs_param) & (abs_param < percent_threshold))] = 1

        # sanity check
        # one_counts = (mask_2d == 1).sum()
        # print(one_counts/param.size)

        layer_to_mask[layer] = mask_2d
        if keep_sign:
            layer_to_mask[layer] *= np.sign(mask_2d)

    return layer_to_mask

def generate_1d_magnitude_binary_mask(top_or_low, model_path, percent=0.2):

    """
        returns 1d binary mask. 
        Compared to generate_2d_magnitude_mask(), this does not keep sign.
    """
    
    layer_to_mask = {}

    with open(model_path, 'rb') as f:
        nn_layer_to_weights = pickle.load(f)
            
    for layer, param in nn_layer_to_weights.items():
    
        param_1d_array = param.flatten()

        # take abs as we show magnitude values
        param_1d_array = np.absolute(param_1d_array)

        pruned_percent = param_1d_array[param_1d_array == 0].size/param_1d_array.size
        
        # if need low, this is a trick
        proxy_percent = percent + pruned_percent if top_or_low == 'low' else percent
            
        percent_order = math.ceil(param_1d_array.size * proxy_percent)

        if top_or_low == 'top':
            percent_threshold = np.partition(param_1d_array, -percent_order)[-percent_order]
        elif top_or_low == 'low':
            percent_threshold = np.partition(param_1d_array, percent_order - 1)[percent_order - 1]

        mask_1d_array = np.empty_like(param_1d_array)
        mask_1d_array[:] = 0 # initialize as 0

        if top_or_low == 'top':
            # change top weights to 1
            mask_1d_array[np.where(param_1d_array >= percent_threshold)] = 1
        elif top_or_low == 'low':
            # change low weights to 1, ingore pruned weights
            mask_1d_array[np.where((0 < param_1d_array) & (param_1d_array < percent_threshold))] = 1
        # sanity check
        # one_counts = (mask_1d_array == 1).sum()
        # print(one_counts/param_1d_array.size)

        layer_to_mask[layer] = mask_1d_array
    return layer_to_mask

def calculate_overlapping_mask(top_or_low, model_paths, percent=0.2):
    layer_to_masks = []

    for model_path in model_paths:
        layer_to_masks.append(generate_2d_magnitude_mask(top_or_low, model_path, percent, keep_sign = False, ignore_pruned = False))

    ref_layer_to_mask = layer_to_masks[0]

    for layer_to_mask in layer_to_masks[1:]:
        for layer, mask in layer_to_mask.items():
            ref_layer_to_mask[layer] *= mask

    return ref_layer_to_mask


def calculate_overlapping_ratio(top_or_low, model_paths, percent=0.2):
    """
    Args:
        model_paths - list of model paths
    """
    layer_to_masks = []

    for model_path in model_paths:
        layer_to_masks.append(generate_1d_magnitude_binary_mask(top_or_low, model_path, percent))

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

def global_vs_local_overlapping(log_folder, comm_round, top_or_low, percent=0.2):
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

    layer_to_clients_to_overlappings = {k: {c: [] for c in clients} for k in layers}

    client_to_xticks = {}
    for client in clients:
        models_folder = f"{log_folder}/models_weights/{client}"
        model_files_this_round = [f for f in os.listdir(models_folder) if os.path.isfile(os.path.join(models_folder, f)) and f"R{comm_round}" in f]
        model_files_this_round.sort(key=lambda x: int(x.split('E')[1].split(".")[0]))
        client_to_xticks[client] = [f.split(".")[0] for f in model_files_this_round]
        
        for epoch_file in model_files_this_round:
            epoch_file_path = f"{log_folder}/models_weights/{client}/{epoch_file}"
            layer_to_ratio = calculate_overlapping_ratio(top_or_low, [global_model_path, epoch_file_path], percent)
            for layer, ratio in layer_to_ratio.items():
                layer_to_clients_to_overlappings[layer][client].append(ratio)
    
    # plot by layer
    """
    1: [4,6,8], # baseline - blue
    2: [4,6,8], # three matches - green
    3: [4,6,7], # two matches - orange
    4: [4,7,8], # two matches, with 0/1 and 2 - orange
    5: [4,7,9], # 1 match - olive
    6: [4,5,9], # 1 match, but two matches with 5 - olive
    7: [3,5,7], # 0 match - grey
    8: [0,1,2], # 0 match - grey
    9: [4,6,8], # three matches, Malicious - red
    10: [4,6,7], # two matches, Malicious - red
    11: [4,5,9], # 1 match, Malicious - red
    12: [3,5,7], # 0 match, Malicious - red
    """
    colors = ["blue", "green", "orange", "orange", "olive", "olive", "grey", "grey", "red", "red", "red", "red"]

    for layer, clients_to_overlappings in layer_to_clients_to_overlappings.items():
        plt.figure(dpi=250)
        iteration = 1
        for client, overlappings in clients_to_overlappings.items():
            plot_y = [percentage * 100 for percentage in overlappings] # used to format percentage
            plt.plot(client_to_xticks[client], plot_y, color=colors[iteration - 1])
            plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(decimals=2))
            # if client[0] == 'L':
            #     plt.plot(client_to_xticks[client], overlappings, color='green')
            # else:
            #     plt.plot(client_to_xticks[client], overlappings, color='red')
            plt.annotate(client, xy=(iteration % len(client_to_xticks[client]), plot_y[iteration % len(plot_y)]), size=8, color=colors[iteration - 1])
            # plt.annotate(client, xy=(len(overlappings) - iteration * 0.8, overlappings[iteration % len(overlappings)]), size=6, color=colors[iteration - 1])
            iteration += 1

        patch_B = mpatches.Patch(color='blue', label='BASE [4, 6, 8]')
        patch_3 = mpatches.Patch(color='green', label=f'3 matches')
        patch_2 = mpatches.Patch(color='orange', label=f'2 matches')
        patch_1 = mpatches.Patch(color='olive', label=f'1 match')
        patch_0 = mpatches.Patch(color='grey', label=f'0 match')
        patch_M = mpatches.Patch(color='red', label=f'Malicious')

        plt.legend(handles=[patch_B, patch_3, patch_2, patch_1, patch_0, patch_M], loc='best')

        plt.xlabel('Epoch')
        plt.ylabel('Overlapping Ratio')
        plt.title(f"Global R{comm_round - 1} vs. Local R{comm_round} - {layer}")

        plot_save_folder = f"{log_folder}/plots/R{comm_round}"
        os.makedirs(plot_save_folder, exist_ok=True)

        plt.savefig(f"{plot_save_folder}/global_vs_local_{layer}.png")


# global_vs_local_overlapping("/Users/chenhang/Documents/Temp/11282022_143103_SEED_50_NOISE_PERCENT_0.2_VAR_1.0", 1, "top", percent=0.2)


def last_local_vs_locals_overlapping(log_folder, ref_client, comm_round, last_epoch, top_or_low, percent=0.2, color_M=True):
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
            layer_to_ratio = calculate_overlapping_ratio(top_or_low, [ref_local_model_path, epoch_file_path], percent)
            for layer, ratio in layer_to_ratio.items():
                layer_to_clients_to_overlappings[layer][client].append(ratio)

    # plot by layer, color by amount of common labels
    matching_colors = {3: "green", 2: "orange", 1: "blue", 0: "grey", "M": "red"}
    ref_client_labels = ast.literal_eval(ref_client.split("_")[1])

    for layer, clients_to_overlappings in layer_to_clients_to_overlappings.items():
        plt.figure(dpi=250)
        iteration = 1
        for client, overlappings in clients_to_overlappings.items():
            plot_y = [percentage * 100 for percentage in overlappings] # used to format percentage
            client_labels = ast.literal_eval(client.split("_")[1])
            num_matching_labels = len(set(ref_client_labels) & set(client_labels))
            if client[0] == "M" and color_M:
                num_matching_labels = "M"
            plt.plot(client_to_xticks[client], plot_y, color=matching_colors[num_matching_labels])
            plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(decimals=2))
            plt.annotate(client, xy=(iteration % len(client_to_xticks[client]), plot_y[iteration % len(plot_y)]), size=8, color=matching_colors[num_matching_labels])
            iteration += 1

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

# last_local_vs_locals_overlapping("/Users/chenhang/Documents/Temp/11262022_011637_SEED_60_NOISE_1.0", "L_[4, 6, 8]_1", 1, 10, "top", percent=0.2, color_M=False)


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

def prune_by_low_overlap(model, last_local_model_paths, prune_threshold, dev_device):

    low_mask = calculate_overlapping_mask("low", last_local_model_paths, percent=1 - prune_threshold)
    params_to_prune = get_prune_params_with_layer_name(model)
    layer_TO_if_pruned = {}
    layer_TO_pruned_percentage = {}
    for params, layer in params_to_prune:
        layer_low_mask = low_mask[layer]
        # reverse the mask
        reversed_mask = np.ones_like(layer_low_mask)
        reversed_mask[layer_low_mask == 1] = 0
        # determine if prune
        old_pruned_percent = float((params.weight == 0).sum()/torch.numel(params.weight))
        new_pruned_percent = float((reversed_mask == 0).sum()/torch.numel(params.weight))
        if old_pruned_percent != new_pruned_percent and new_pruned_percent < prune_threshold:
            lowOverlappingPrune(params, reversed_mask, dev_device)
            layer_TO_if_pruned[layer] = True
            layer_TO_pruned_percentage[layer] = new_pruned_percent
        else:
            layer_TO_if_pruned[layer] = False
            layer_TO_pruned_percentage[layer] = old_pruned_percent

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