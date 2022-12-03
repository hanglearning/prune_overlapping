import numpy as np
from torchvision import datasets, transforms


def get_dataset_mnist_extr_noniid(num_users, n_class, nsamples, rate_unbalance, log_dirpath, deterministic_sharding):
    data_dir = './data'
    apply_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                   transform=apply_transform)

    test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                  transform=apply_transform)

    # Chose euqal splits for every user
    user_groups_train, user_groups_test, user_groups_labels = mnist_extr_noniid(
        train_dataset, test_dataset, num_users, n_class, nsamples, rate_unbalance, log_dirpath, deterministic_sharding)
    return train_dataset, test_dataset, user_groups_train, user_groups_test, user_groups_labels


def mnist_extr_noniid(train_dataset, test_dataset, num_users, n_class, num_samples, rate_unbalance, log_dirpath, deterministic_sharding):
    num_shards_train, num_imgs_train = int(60000/num_samples), num_samples
    num_classes = 10
    num_imgs_perc_test, num_imgs_test_total = 1000, 10000

    assert(n_class * num_users <= num_shards_train)
    assert(n_class <= num_classes)

    idx_class = [i for i in range(num_classes)]
    idx_shard = np.array([i for i in range(num_shards_train)])
    dict_users_train = {i: np.array([]) for i in range(num_users)}
    dict_users_test = {i: np.array([]) for i in range(num_users)}
    dict_users_labels = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards_train*num_imgs_train)

    labels = np.array(train_dataset.targets)
    idxs_test = np.arange(num_imgs_test_total)
    labels_test = np.array(test_dataset.targets)

    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    labels = idxs_labels[1, :]

    idxs_labels_test = np.vstack((idxs_test, labels_test))
    idxs_labels_test = idxs_labels_test[:, idxs_labels_test[1, :].argsort()]
    idxs_test = idxs_labels_test[0, :]
    labels_test = idxs_labels_test[1, :]

    idxs_test_splits = [[] for i in range(num_classes)]
    for i in range(len(labels_test)):
        idxs_test_splits[labels_test[i]].append(idxs_test[i])

    idx_shards = np.split(idx_shard, 10)

    # deterministic labels to verify fc2, set users to 12
    user_to_labels = {0: [4,6,8], # baseline
                      1: [4,6,8], # three matches
                      2: [4,6,7], # two matches
                      3: [4,7,8], # two matches, with 0/1 and 2
                      4: [4,7,9], # 1 match
                      5: [4,5,9], # 1 match, but two matches with 5
                      6: [3,5,7], # 0 match
                      7: [0,1,2], # 0 match
                      8: [4,6,8], # three matches, Malicious
                      9: [4,6,7], # two matches, Malicious
                      10: [4,5,9], # 1 match, Malicious
                      11: [3,5,7], # 0 match, Malicious
                      }

    for i in range(num_users):
        user_labels = np.array([])
        temp_set = list(set(np.random.choice(10, n_class, replace=False)))
        if deterministic_sharding and i in user_to_labels.keys():
            temp_set = user_to_labels[i]
        dict_users_labels[i] = temp_set
        rand_set = []
        for j in temp_set:
            choice = np.random.choice(idx_shards[j], 1)[0]
            rand_set.append(int(choice))
            idx_shards[j] = np.delete(
                idx_shards[j], np.where(idx_shards[j] == choice))
        unbalance_flag = 0
        label_to_qty = {}
        for rand_iter in range(len(rand_set)):
            rand = rand_set[rand_iter]
            if unbalance_flag == 0:
                dict_users_train[i] = np.concatenate(
                    (dict_users_train[i], idxs[rand*num_imgs_train:(rand+1)*num_imgs_train]), axis=0)
                label_to_qty[temp_set[rand_iter]] = len(idxs[rand*num_imgs_train:(rand+1)*num_imgs_train])
                user_labels = np.concatenate(
                    (user_labels, labels[rand*num_imgs_train:(rand+1)*num_imgs_train]), axis=0)
            else:
                dict_users_train[i] = np.concatenate(
                    (dict_users_train[i], idxs[rand*num_imgs_train:int((rand+rate_unbalance)*num_imgs_train)]), axis=0)
                label_to_qty[temp_set[rand_iter]] = len(idxs[rand*num_imgs_train:int((rand+rate_unbalance)*num_imgs_train)])
                user_labels = np.concatenate(
                    (user_labels, labels[rand*num_imgs_train:int((rand+rate_unbalance)*num_imgs_train)]), axis=0)
            unbalance_flag = 1
        

        display_text = f"Client {i+1}  - labels {list(label_to_qty.keys())}, corresponding qty {list(label_to_qty.values())}"
        with open(f'{log_dirpath}/dataset_assigned.txt', 'a') as f:
            f.write(f'{display_text}\n')
        print(display_text)
            
        user_labels_set = set(user_labels)

        for label in user_labels_set:
            dict_users_test[i] = np.concatenate(
                (dict_users_test[i], idxs_test_splits[int(label)]), axis=0)
    return dict_users_train, dict_users_test, dict_users_labels
