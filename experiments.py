import numpy as np
import json
import torch
import torch.optim as optim
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.utils.data as data
import argparse
import logging
import os
import copy
from math import *
import random

import datetime
#from torch.utils.tensorboard import SummaryWriter

from model import *
from utils import *
from vggmodel import *
from resnetcifar import *

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='mlp', help='neural network used in training')
    parser.add_argument('--dataset', type=str, default='mnist', help='dataset used for training')
    parser.add_argument('--net_config', type=lambda x: list(map(int, x.split(', '))))
    parser.add_argument('--partition', type=str, default='homo', help='the data partitioning strategy')
    parser.add_argument('--batch-size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate (default: 0.01)')
    parser.add_argument('--epochs', type=int, default=5, help='number of local epochs')
    parser.add_argument('--n_parties', type=int, default=2,  help='number of workers in a distributed cluster')
    parser.add_argument('--alg', type=str, default='fedavg',
                            help='fl algorithms: fedavg/fedprox/scaffold/fednova/moon')
    parser.add_argument('--use_projection_head', type=bool, default=False, help='whether add an additional header to model or not (see MOON)')
    parser.add_argument('--out_dim', type=int, default=256, help='the output dimension for the projection layer')
    parser.add_argument('--loss', type=str, default='contrastive', help='for moon')
    parser.add_argument('--temperature', type=float, default=0.5, help='the temperature parameter for contrastive loss')
    parser.add_argument('--comm_round', type=int, default=50, help='number of maximum communication roun')
    parser.add_argument('--is_same_initial', type=int, default=1, help='Whether initial all the models with the same parameters in fedavg')
    parser.add_argument('--init_seed', type=int, default=0, help="Random seed")
    parser.add_argument('--dropout_p', type=float, required=False, default=0.0, help="Dropout probability. Default=0.0")
    parser.add_argument('--datadir', type=str, required=False, default="./data/", help="Data directory")
    parser.add_argument('--reg', type=float, default=1e-5, help="L2 regularization strength")
    parser.add_argument('--logdir', type=str, required=False, default="./logs/", help='Log directory path')
    parser.add_argument('--modeldir', type=str, required=False, default="./models/", help='Model directory path')
    parser.add_argument('--beta', type=float, default=0.5, help='The parameter for the dirichlet distribution for data partitioning')
    parser.add_argument('--device', type=str, default='cuda:0', help='The device to run the program')
    parser.add_argument('--log_file_name', type=str, default=None, help='The log file name')
    parser.add_argument('--optimizer', type=str, default='sgd', help='the optimizer')
    parser.add_argument('--mu', type=float, default=0.001, help='the mu parameter for fedprox')
    parser.add_argument('--noise', type=float, default=0, help='how much noise we add to some party')
    parser.add_argument('--noise_type', type=str, default='level', help='Different level of noise or different space of noise')
    parser.add_argument('--rho', type=float, default=0, help='Parameter controlling the momentum SGD')
    parser.add_argument('--sample', type=float, default=1, help='Sample ratio for each communication round')
    parser.add_argument('--p', type=float, default=0.8, help="Proportion of regular clients")
    parser.add_argument('--q', type=float, default=0.2, help="Proportion of clients doing partial updates")
    args = parser.parse_args()
    return args

def init_nets(net_configs, dropout_p, n_parties, args):

    nets = {net_i: None for net_i in range(n_parties)}

    if args.dataset in {'mnist', 'cifar10', 'svhn', 'fmnist'}:
        n_classes = 10
    elif args.dataset == 'celeba':
        n_classes = 2
    elif args.dataset == 'cifar100':
        n_classes = 100
    elif args.dataset == 'tinyimagenet':
        n_classes = 200
    elif args.dataset == 'femnist':
        n_classes = 62
    elif args.dataset == 'emnist':
        n_classes = 47
    elif args.dataset in {'a9a', 'covtype', 'rcv1', 'SUSY'}:
        n_classes = 2
    if args.use_projection_head:
        add = ""
        if "mnist" in args.dataset and args.model == "simple-cnn":
            add = "-mnist"
        for net_i in range(n_parties):
            net = ModelFedCon(args.model+add, args.out_dim, n_classes, net_configs)
            nets[net_i] = net
    else:
        if args.alg == 'moon':
            add = ""
            if "mnist" in args.dataset and args.model == "simple-cnn":
                add = "-mnist"
            for net_i in range(n_parties):
                net = ModelFedCon_noheader(args.model+add, args.out_dim, n_classes, net_configs)
                nets[net_i] = net
        else:
            for net_i in range(n_parties):
                if args.dataset == "generated":
                    net = PerceptronModel()
                elif args.model == "mlp":
                    if args.dataset == 'covtype':
                        input_size = 54
                        output_size = 2
                        hidden_sizes = [32,16,8]
                    elif args.dataset == 'a9a':
                        input_size = 123
                        output_size = 2
                        hidden_sizes = [32,16,8]
                    elif args.dataset == 'rcv1':
                        input_size = 47236
                        output_size = 2
                        hidden_sizes = [32,16,8]
                    elif args.dataset == 'SUSY':
                        input_size = 18
                        output_size = 2
                        hidden_sizes = [16,8]
                    net = FcNet(input_size, hidden_sizes, output_size, dropout_p)
                elif args.model == "vgg":
                    net = vgg11()
                elif args.model == "simple-cnn":
                    if args.dataset in ("cifar10", "cinic10", "svhn"):
                        net = SimpleCNN(input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=10)
                    elif args.dataset in ("mnist", 'femnist', 'fmnist'):
                        net = SimpleCNNMNIST(input_dim=(16 * 4 * 4), hidden_dims=[120, 84], output_dim=10)
                    elif args.dataset == 'celeba':
                        net = SimpleCNN(input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=2)
                elif args.model == "vgg-9":
                    if args.dataset in ("mnist", 'femnist'):
                        net = ModerateCNNMNIST()
                    elif args.dataset in ("cifar10", "cinic10", "svhn"):
                        # print("in moderate cnn")
                        net = ModerateCNN()
                    elif args.dataset == 'celeba':
                        net = ModerateCNN(output_dim=2)
                elif args.model == "resnet":
                    net = ResNet50_cifar10()
                elif args.model == "vgg16":
                    net = vgg16()
                else:
                    print("not supported yet")
                    exit(1)
                nets[net_i] = net

    model_meta_data = []
    layer_type = []
    for (k, v) in nets[0].state_dict().items():
        model_meta_data.append(v.shape)
        layer_type.append(k)
    return nets, model_meta_data, layer_type


def train_net(net_id, net, train_dataloader, test_dataloader, epochs, lr, args_optimizer, device="cpu"):
    logger.info('Training network %s' % str(net_id))

    # Initial accuracy before training
    train_acc = compute_accuracy(net, train_dataloader, device=device)
    test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    logger.info('>> Pre-Training Training accuracy: {}'.format(train_acc))
    logger.info('>> Pre-Training Test accuracy: {}'.format(test_acc))

    # Choose the optimizer
    if args_optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
    elif args_optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg, amsgrad=True)
    elif args_optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=args.rho, weight_decay=args.reg)

    # Define the loss function
    criterion = nn.CrossEntropyLoss().to(device)

    cnt = 0
    if type(train_dataloader) == type([1]):
        pass
    else:
        train_dataloader = [train_dataloader]

    # Training loop
    total_loss_collector = []  # To store total loss across epochs
    for epoch in range(epochs):
        epoch_loss_collector = []  # To store loss for each batch in the epoch
        for tmp in train_dataloader:
            for batch_idx, (x, target) in enumerate(tmp):
                x, target = x.to(device), target.to(device)

                # Zero gradients and set input/output for forward pass
                optimizer.zero_grad()
                x.requires_grad = True
                target.requires_grad = False
                target = target.long()

                # Forward pass and calculate loss
                out = net(x)
                loss = criterion(out, target)

                # Backward pass and optimizer step
                loss.backward()
                optimizer.step()

                # Collect loss for the epoch
                epoch_loss_collector.append(loss.item())

        # Calculate the average loss for the epoch
        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        total_loss_collector.append(epoch_loss)  # Store for round-averaged loss
        logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))

    # Calculate the average training loss for the round
    avg_training_loss = sum(total_loss_collector) / len(total_loss_collector)
    logger.info('>> Average Training Loss for Network %d: %f' % (net_id, avg_training_loss))

    # Final evaluation after training
    train_acc = compute_accuracy(net, train_dataloader, device=device)
    test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    logger.info('>> Training accuracy: %f' % train_acc)
    logger.info('>> Test accuracy: %f' % test_acc)

    # Move model back to CPU and log completion
    net.to('cpu')
    logger.info(' ** Training complete **')
    return train_acc, test_acc, avg_training_loss


def train_net_scaffold(net_id, net, global_model, c_local, c_global, train_dataloader, test_dataloader, epochs, lr, args_optimizer, device="cpu"):
    logger.info('Training network %s' % str(net_id))

    train_acc = compute_accuracy(net, train_dataloader, device=device)
    test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    logger.info('>> Pre-Training Training accuracy: {}'.format(train_acc))
    logger.info('>> Pre-Training Test accuracy: {}'.format(test_acc))

    if args_optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
    elif args_optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg,
                               amsgrad=True)
    elif args_optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=args.rho, weight_decay=args.reg)
    criterion = nn.CrossEntropyLoss().to(device)

    cnt = 0
    if type(train_dataloader) == type([1]):
        pass
    else:
        train_dataloader = [train_dataloader]

    #writer = SummaryWriter()

    c_local.to(device)
    c_global.to(device)
    global_model.to(device)

    c_global_para = c_global.state_dict()
    c_local_para = c_local.state_dict()

    for epoch in range(epochs):
        epoch_loss_collector = []
        for tmp in train_dataloader:
            for batch_idx, (x, target) in enumerate(tmp):
                x, target = x.to(device), target.to(device)

                optimizer.zero_grad()
                x.requires_grad = True
                target.requires_grad = False
                target = target.long()

                out = net(x)
                loss = criterion(out, target)

                loss.backward()
                optimizer.step()

                net_para = net.state_dict()
                for key in net_para:
                    net_para[key] = net_para[key] - args.lr * (c_global_para[key] - c_local_para[key])
                net.load_state_dict(net_para)

                cnt += 1
                epoch_loss_collector.append(loss.item())


        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))

    c_new_para = c_local.state_dict()
    c_delta_para = copy.deepcopy(c_local.state_dict())
    global_model_para = global_model.state_dict()
    net_para = net.state_dict()
    for key in net_para:
        c_new_para[key] = c_new_para[key] - c_global_para[key] + (global_model_para[key] - net_para[key]) / (cnt * args.lr)
        c_delta_para[key] = c_new_para[key] - c_local_para[key]
    c_local.load_state_dict(c_new_para)


    train_acc = compute_accuracy(net, train_dataloader, device=device)
    test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    logger.info('>> Training accuracy: %f' % train_acc)
    logger.info('>> Test accuracy: %f' % test_acc)

    net.to('cpu')
    logger.info(' ** Training complete **')
    return train_acc, test_acc, c_delta_para


def view_image(train_dataloader):
    for (x, target) in train_dataloader:
        np.save("img.npy", x)
        print(x.shape)
        exit(0)


def local_train_net(nets, selected, args, net_dataidx_map, test_dl=None, device="cpu"):
    """
    Trains the local models for the selected clients and computes average metrics across clients.

    Args:
        nets (dict): Dictionary of client models.
        selected (list): List of selected client indices for the current round.
        args (Namespace): Arguments containing experiment configurations.
        net_dataidx_map (dict): Mapping of client indices to their data indices.
        test_dl (DataLoader): Test DataLoader for evaluation (optional).
        device (str): Device to use for training (e.g., "cpu" or "cuda").

    Returns:
        list: List of updated client models.
    """
    avg_acc = 0.0  # Variable to store the average test accuracy.
    total_train_loss = 0.0  # Variable to store the total training loss.
    total_clients = len(selected)  # Number of selected clients.

    # Determine the number of clients in `p` and `q` groups
    num_clients_p = int(args.n_parties * args.p)
    num_clients_q = int(args.n_parties * args.q)

    # Randomly split the selected clients into `p` and `q` groups
    np.random.shuffle(selected)
    clients_p = selected[:num_clients_p]
    clients_q = selected[num_clients_p:]
    print("Number of participating clients: %d" % total_clients)
    print("Number of clients in group `p`: %d" % len(clients_p))
    print("Number of clients in group `q`: %d" % len(clients_q))
    
    # Loop through each client model in the dictionary.
    for net_id, net in nets.items():
        # Skip clients that are not selected for the current round.
        if net_id not in selected:
            continue

        # Retrieve the data indices assigned to this client.
        dataidxs = net_dataidx_map[net_id]

        # Log the training information for this client.
        logger.info("Training network %s. n_training: %d" % (str(net_id), len(dataidxs)))

        # Move the model to the specified device (e.g., GPU).
        net.to(device)

        # Determine noise level for this client.
        noise_level = args.noise
        if net_id == args.n_parties - 1:  # Special case for the last client.
            noise_level = 0

        # Get the DataLoader for this client's data with noise configuration.
        if args.noise_type == 'space':  # If noise type is spatial.
            train_dl_local, test_dl_local, _, _ = get_dataloader(
                args.dataset, args.datadir, args.batch_size, 32, dataidxs, noise_level, net_id, args.n_parties - 1
            )
        else:  # If noise type is level-based.
            noise_level = args.noise / (args.n_parties - 1) * net_id
            train_dl_local, test_dl_local, _, _ = get_dataloader(
                args.dataset, args.datadir, args.batch_size, 32, dataidxs, noise_level
            )

        # Get global DataLoader for comparison or reference (if needed).
        train_dl_global, test_dl_global, _, _ = get_dataloader(
            args.dataset, args.datadir, args.batch_size, 32
        )

        def truncated_poisson(lam, min_val=1):
            while True:
                sample = np.random.poisson(lam)
                if sample >= min_val:
                    return sample
        
        # Determine the number of epochs for training.
        if net_id in clients_q:  # Partial updates for clients in group `q`
            n_epoch = min(truncated_poisson(args.epochs), args.epochs)
            print("Client %d performing partial updates with %d epochs" % (net_id, n_epoch))
        else:  # Full updates for clients in group `p`
            n_epoch = args.epochs
            print("Client %d performing full updates with %d epochs" % (net_id, n_epoch))

        # Train the client's model and evaluate its performance.
        trainacc, testacc, train_loss = train_net(
            net_id, net, train_dl_local, test_dl, n_epoch, args.lr, args.optimizer, device=device
        )
        logger.info("net %d final test acc %f" % (net_id, testacc))

        # Accumulate metrics for averaging later.
        avg_acc += testacc
        total_train_loss += train_loss

    # Compute the average metrics across all selected clients.
    avg_acc /= total_clients
    avg_train_loss = total_train_loss / total_clients

    # Log the average metrics.
    logger.info("Round Average Training Loss: %f" % avg_train_loss)
    logger.info("Round Average Test Accuracy: %f" % avg_acc)

    # Return the updated list of client models.
    nets_list = list(nets.values())
    return nets_list


def local_train_net_scaffold(nets, selected, global_model, c_nets, c_global, args, net_dataidx_map, test_dl = None, device="cpu"):
    avg_acc = 0.0

    total_delta = copy.deepcopy(global_model.state_dict())
    for key in total_delta:
        total_delta[key] = 0.0
    c_global.to(device)
    global_model.to(device)
    for net_id, net in nets.items():
        if net_id not in selected:
            continue
        dataidxs = net_dataidx_map[net_id]

        logger.info("Training network %s. n_training: %d" % (str(net_id), len(dataidxs)))
        # move the model to cuda device:
        net.to(device)

        c_nets[net_id].to(device)

        noise_level = args.noise
        if net_id == args.n_parties - 1:
            noise_level = 0

        if args.noise_type == 'space':
            train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs, noise_level, net_id, args.n_parties-1)
        else:
            noise_level = args.noise / (args.n_parties - 1) * net_id
            train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs, noise_level)
        train_dl_global, test_dl_global, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32)
        n_epoch = args.epochs


        trainacc, testacc, c_delta_para = train_net_scaffold(net_id, net, global_model, c_nets[net_id], c_global, train_dl_local, test_dl, n_epoch, args.lr, args.optimizer, device=device)

        c_nets[net_id].to('cpu')
        for key in total_delta:
            total_delta[key] += c_delta_para[key]


        logger.info("net %d final test acc %f" % (net_id, testacc))
        avg_acc += testacc
    for key in total_delta:
        total_delta[key] /= args.n_parties
    c_global_para = c_global.state_dict()
    for key in c_global_para:
        if c_global_para[key].type() == 'torch.LongTensor':
            c_global_para[key] += total_delta[key].type(torch.LongTensor)
        elif c_global_para[key].type() == 'torch.cuda.LongTensor':
            c_global_para[key] += total_delta[key].type(torch.cuda.LongTensor)
        else:
            #print(c_global_para[key].type())
            c_global_para[key] += total_delta[key]
    c_global.load_state_dict(c_global_para)

    avg_acc /= len(selected)
    if args.alg == 'local_training':
        logger.info("avg test acc %f" % avg_acc)

    nets_list = list(nets.values())
    return nets_list

def get_partition_dict(dataset, partition, n_parties, init_seed=0, datadir='./data', logdir='./logs', beta=0.5):
    seed = init_seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = partition_data(
        dataset, datadir, logdir, partition, n_parties, beta=beta)

    return net_dataidx_map

if __name__ == '__main__':
    # Get the command-line arguments
    args = get_args() 

    # Create the log and model directories if they don't already exist
    mkdirs(args.logdir) 
    mkdirs(args.modeldir) 

    # Save the experiment arguments to the log directory for reference
    if args.log_file_name is None: 
        argument_path = 'experiment_arguments-%s.json' % datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    else:
        argument_path = args.log_file_name + '.json'        
    with open(os.path.join(args.logdir, argument_path), 'w') as f:
        json.dump(str(args), f)
            
    # Set the device (e.g., 'cuda:0' for GPU or 'cpu')
    device = torch.device(args.device) 

    # Set up the logger to log experiment details to a file
    for handler in logging.root.handlers[:]: 
        logging.root.removeHandler(handler)

    # Define the log file name if not provided
    if args.log_file_name is None: 
        args.log_file_name = 'experiment_log-%s' % (datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S"))        
    log_path = args.log_file_name + '.log' 

    # Configure logging settings
    logging.basicConfig( 
        filename=os.path.join(args.logdir, log_path),  # Log file path
        format='%(asctime)s %(levelname)-8s %(message)s',  # Log format
        datefmt='%m-%d %H:%M',  # Date and time format
        level=logging.DEBUG,  # Logging level
        filemode='w'  # Overwrite existing log file
    )
    logger = logging.getLogger() 
    logger.setLevel(logging.DEBUG)
    logger.info(device)

    # Set the random seed for reproducibility
    seed = args.init_seed 
    logger.info("#" * 100)  # Log a separator for better readability
    np.random.seed(seed)  # Set seed for NumPy
    torch.manual_seed(seed)  # Set seed for PyTorch
    random.seed(seed)  # Set seed for Python's random module
        
    # Partition the dataset according to the specified strategy
    logger.info("Partitioning data")
    X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = partition_data(
        args.dataset, args.datadir, args.logdir, args.partition, args.n_parties, beta=args.beta
    )

    # Determine the number of classes in the dataset
    n_classes = len(np.unique(y_train))

    # Get dataloaders for the global model (training and testing)
    train_dl_global, test_dl_global, train_ds_global, test_ds_global = get_dataloader(
        args.dataset,  # Dataset name (e.g., cifar10, mnist)
        args.datadir,  # Directory where data is stored
        args.batch_size,  # Batch size for loading data
        32  # Additional argument for the dataloader (e.g., shuffle buffer size)
    )
    print("len train_dl_global:", len(train_ds_global))  # Print the size of the global training dataset
    data_size = len(test_ds_global)  # Determine the size of the global testing dataset

    # Initialize lists to collect datasets for training and testing across all parties
    train_all_in_list = []
    test_all_in_list = []

    # Check if noise is applied to the data
    if args.noise > 0:
        # Loop through each party (client) in the federated learning setup
        for party_id in range(args.n_parties):
            # Get the indices of the data samples assigned to this party
            dataidxs = net_dataidx_map[party_id]

            # Set the noise level for this party
            noise_level = args.noise
            # If this is the last party, no noise is added (noise level = 0)
            if party_id == args.n_parties - 1:
                noise_level = 0

            # Handle different types of noise
            if args.noise_type == 'space':
                # If noise type is 'space', apply noise differently based on the party
                train_dl_local, test_dl_local, train_ds_local, test_ds_local = get_dataloader(
                    args.dataset, args.datadir, args.batch_size, 32, dataidxs, 
                    noise_level, party_id, args.n_parties - 1
                )
            else:
                # Otherwise, distribute noise proportionally among parties
                noise_level = args.noise / (args.n_parties - 1) * party_id
                train_dl_local, test_dl_local, train_ds_local, test_ds_local = get_dataloader(
                    args.dataset, args.datadir, args.batch_size, 32, dataidxs, noise_level
                )
            
            # Collect the training and testing datasets for this party
            train_all_in_list.append(train_ds_local)
            test_all_in_list.append(test_ds_local)

        # Combine the training datasets from all parties into a single dataset
        train_all_in_ds = data.ConcatDataset(train_all_in_list)
        # Create a DataLoader for the combined training dataset, with shuffling enabled
        train_dl_global = data.DataLoader(dataset=train_all_in_ds, batch_size=args.batch_size, shuffle=True)

        # Combine the testing datasets from all parties into a single dataset
        test_all_in_ds = data.ConcatDataset(test_all_in_list)
        # Create a DataLoader for the combined testing dataset, with shuffling disabled
        test_dl_global = data.DataLoader(dataset=test_all_in_ds, batch_size=32, shuffle=False)



    if args.alg == 'fedavg':
        # Log the start of FedAvg algorithm initialization
        logger.info("Initializing nets")
        
        # Initialize local models (nets) and metadata for all parties
        nets, local_model_meta_data, layer_type = init_nets(args.net_config, args.dropout_p, args.n_parties, args)
        
        # Initialize the global model (shared among all parties)
        global_models, global_model_meta_data, global_layer_type = init_nets(args.net_config, 0, 1, args)
        global_model = global_models[0]

        # Retrieve the global model's parameters
        global_para = global_model.state_dict()

        # If all local models should start with the same parameters as the global model
        if args.is_same_initial:
            for net_id, net in nets.items():
                net.load_state_dict(global_para)

        # Main communication rounds loop
        for round in range(args.comm_round):
            # Log the current communication round
            logger.info("in comm round:" + str(round))

            # Randomly select a subset of parties for this communication round
            arr = np.arange(args.n_parties)
            np.random.shuffle(arr)
            selected = arr[:int(args.n_parties * args.sample)]

            # Load the global model parameters into the selected local models
            global_para = global_model.state_dict()
            if round == 0:
                if args.is_same_initial:
                    for idx in selected:
                        nets[idx].load_state_dict(global_para)
            else:
                for idx in selected:
                    nets[idx].load_state_dict(global_para)

            # Train the selected local models
            local_train_net(nets, selected, args, net_dataidx_map, test_dl=test_dl_global, device=device)

            # Aggregate local updates to update the global model using Federated Averaging
            total_data_points = sum([len(net_dataidx_map[r]) for r in selected])
            fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in selected]

            # Compute the weighted average of the selected local model parameters
            for idx in range(len(selected)):
                net_para = nets[selected[idx]].cpu().state_dict()
                if idx == 0:
                    for key in net_para:
                        global_para[key] = net_para[key] * fed_avg_freqs[idx]
                else:
                    for key in net_para:
                        global_para[key] += net_para[key] * fed_avg_freqs[idx]
            
            # Update the global model with the aggregated parameters
            global_model.load_state_dict(global_para)

            # Log dataset statistics for debugging
            logger.info('global n_training: %d' % len(train_dl_global))
            logger.info('global n_test: %d' % len(test_dl_global))

            # Evaluate the global model's performance on the training and test datasets
            global_model.to(device)
            train_acc = compute_accuracy(global_model, train_dl_global, device=device)
            test_acc, conf_matrix = compute_accuracy(global_model, test_dl_global, get_confusion_matrix=True, device=device)

            # Log the global model's training and test accuracy for the current round
            logger.info('>> Global Model Train accuracy: %f' % train_acc)
            logger.info('>> Global Model Test accuracy: %f' % test_acc)


    elif args.alg == 'scaffold':
        logger.info("Initializing nets")
        nets, local_model_meta_data, layer_type = init_nets(args.net_config, args.dropout_p, args.n_parties, args)
        global_models, global_model_meta_data, global_layer_type = init_nets(args.net_config, 0, 1, args)
        global_model = global_models[0]

        c_nets, _, _ = init_nets(args.net_config, args.dropout_p, args.n_parties, args)
        c_globals, _, _ = init_nets(args.net_config, 0, 1, args)
        c_global = c_globals[0]
        c_global_para = c_global.state_dict()
        for net_id, net in c_nets.items():
            net.load_state_dict(c_global_para)

        global_para = global_model.state_dict()
        if args.is_same_initial:
            for net_id, net in nets.items():
                net.load_state_dict(global_para)


        for round in range(args.comm_round):
            logger.info("in comm round:" + str(round))

            arr = np.arange(args.n_parties)
            np.random.shuffle(arr)
            selected = arr[:int(args.n_parties * args.sample)]

            global_para = global_model.state_dict()
            if round == 0:
                if args.is_same_initial:
                    for idx in selected:
                        nets[idx].load_state_dict(global_para)
            else:
                for idx in selected:
                    nets[idx].load_state_dict(global_para)

            local_train_net_scaffold(nets, selected, global_model, c_nets, c_global, args, net_dataidx_map, test_dl = test_dl_global, device=device)
            # local_train_net(nets, args, net_dataidx_map, local_split=False, device=device)

            # update global model
            total_data_points = sum([len(net_dataidx_map[r]) for r in selected])
            fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in selected]

            for idx in range(len(selected)):
                net_para = nets[selected[idx]].cpu().state_dict()
                if idx == 0:
                    for key in net_para:
                        global_para[key] = net_para[key] * fed_avg_freqs[idx]
                else:
                    for key in net_para:
                        global_para[key] += net_para[key] * fed_avg_freqs[idx]
            global_model.load_state_dict(global_para)


            logger.info('global n_training: %d' % len(train_dl_global))
            logger.info('global n_test: %d' % len(test_dl_global))

            global_model.to(device)
            train_acc = compute_accuracy(global_model, train_dl_global, device=device)
            test_acc, conf_matrix = compute_accuracy(global_model, test_dl_global, get_confusion_matrix=True, device=device)

            logger.info('>> Global Model Train accuracy: %f' % train_acc)
            logger.info('>> Global Model Test accuracy: %f' % test_acc)

    elif args.alg == 'local_training':
        logger.info("Initializing nets")
        nets, local_model_meta_data, layer_type = init_nets(args.net_config, args.dropout_p, args.n_parties, args)
        arr = np.arange(args.n_parties)
        local_train_net(nets, arr, args, net_dataidx_map, test_dl = test_dl_global, device=device)

    elif args.alg == 'all_in':
        nets, local_model_meta_data, layer_type = init_nets(args.net_config, args.dropout_p, 1, args)
        n_epoch = args.epochs
        nets[0].to(device)
        trainacc, testacc = train_net(0, nets[0], train_dl_global, test_dl_global, n_epoch, args.lr, args.optimizer, device=device)

        logger.info("All in test acc: %f" % testacc)

   
