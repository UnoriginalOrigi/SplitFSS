import argparse
import os
import signal
import subprocess
import time
import pickle

import torch
import torch.optim as optim
import numpy as np

torch.set_num_threads(1)

import syft as sy
from syft.serde.compression import NO_COMPRESSION
from syft.grid.clients.data_centric_fl_client import DataCentricFLClient

sy.serde.compression.default_compress_scheme = NO_COMPRESSION

from procedure import train, test
from data import get_data_loaders, get_number_classes
from models import get_model, load_state_dict
from preprocess import build_prepocessing

def dataset_shuffle(dataset_pub, dataset_priv):
    N = len(dataset_pub)
    indices = np.arange(N)
    indices = np.random.permutation(indices)

    new_dataset_pub = []
    new_dataset_priv = []

    for i in range(0, N):
        new_dataset_pub.append(dataset_pub[indices[i]])
        new_dataset_priv.append(dataset_priv[indices[i]])
    return new_dataset_pub, new_dataset_priv

def run(args):
    if args.train:
        print(f"Training over {args.epochs} epochs")
    elif args.test:
        print("Running a full evaluation")
    else:
        print("Running inference speed test")
    print("model:\t\t", args.model)
    print("dataset:\t", args.dataset)
    print("batch_size:\t", args.batch_size)

    hook = sy.TorchHook(torch)
    
    bob = sy.VirtualWorker(hook, id="bob")
    alice = sy.VirtualWorker(hook, id="alice")
    crypto_provider = sy.VirtualWorker(hook, id="crypto_provider")

    workers = [alice, bob]
    sy.local_worker.clients = workers

    encryption_kwargs = dict(
        workers=workers, crypto_provider=crypto_provider, protocol=args.protocol
    )
    kwargs = dict(
        requires_grad=args.requires_grad,
        precision_fractional=args.precision_fractional,
        dtype=args.dtype,
        **encryption_kwargs,
    )

    public_train_loader, private_train_loader, public_test_loader, private_test_loader = get_data_loaders(args, kwargs, private=True)

    #MODEL SPLIT INTO PRIVATE AND PUBLIC
    if args.model == "split":
        modelPub = get_model("split_pub", args.dataset, out_features=get_number_classes(args.dataset))
        modelPriv = get_model("split_priv", args.dataset, out_features=get_number_classes(args.dataset))
    elif args.model == "usplit":
        modelPub = get_model("split_pub", args.dataset, out_features=get_number_classes(args.dataset))
        modelPriv = get_model(args.model, args.dataset, out_features=get_number_classes(args.dataset))
    else:
        modelPub = None
        modelPriv = get_model("full", args.dataset, out_features=get_number_classes(args.dataset))

    if args.test and not args.train:
        load_state_dict(model, args.model, args.dataset)
    if not args.model == "full":
        modelPub.eval()
        modelPriv.eval()
    else:
        modelPriv.eval()

    if torch.cuda.is_available():
        sy.cuda_force = False

    #UPDATED ENCRYPTION OF MODEL
    if not args.public:
        modelPriv.encrypt(**kwargs)
        if args.fp_only: 
            modelPriv.get()

    model_comms = 0
    client_server_comms = 0
    server_client_comms = 0

    if args.comm_info:
        if not args.public:
            for param in modelPriv.parameters():
                alice_model_shares = param.child.child.child.child['alice'].copy().get()
                bob_model_shares = param.child.child.child.child['bob'].copy().get()
                
                alice_model_comms = len(pickle.dumps(alice_model_shares))
                bob_model_comms = len(pickle.dumps(bob_model_shares))
                model_comms += alice_model_comms + bob_model_comms
        else:
            for param in modelPriv.parameters():              
                model_comms = len(pickle.dumps(param))
                model_comms += model_comms

        print(f"Sent model parameters total: {model_comms} bytes")


    if args.train:
        tot_train_time = 0
        comms_total = []
        test_comms_total = []
        for epoch in range(args.epochs):
            comms_count = 0
            if not args.model == "full":
                optimizerPriv = optim.SGD(modelPriv.parameters(), lr=args.lr, momentum=args.momentum)
                optimizerPub = optim.SGD(modelPub.parameters(), lr=args.lr, momentum=args.momentum)
            else:
                optimizerPriv = optim.SGD(modelPriv.parameters(), lr=args.lr, momentum=args.momentum)
                optimizerPub = None

            if not args.public:
                optimizerPriv = optimizerPriv.fix_precision(
                    precision_fractional=args.precision_fractional, dtype=args.dtype
                )
            #TRAIN() PASSES EXTRA VARIABLES, NAMELY THE SPLIT MODELS modelPub AND modelPriv; kwargs AND SPLIT OPTIMIZERS optimizerPriv and optimizerPub (need more tests if this is needed),
            train_time, comms_count, client_server_comms, server_client_comms = train(args, kwargs, modelPub, modelPriv, private_train_loader,
                                                                                    public_train_loader, optimizerPub, optimizerPriv, comms_count, client_server_comms,
                                                                                    server_client_comms, epoch)
            tot_train_time += train_time

            if args.comm_info:
                comms_total.append(comms_count)
                print(f"Epoch's training time: {train_time} s")
                print(f"Epoch's client communication overhead: {client_server_comms/(epoch+1)} bytes")
                print(f"Epoch's server communication overhead: {server_client_comms/(epoch+1)} bytes")
                print(f"Epoch's total communication overhead: {comms_count} bytes")
                
            public_train_loader, private_train_loader = dataset_shuffle(public_train_loader, private_train_loader)

            test_time, test_comms = test(args, kwargs, modelPub, modelPriv, private_test_loader, public_test_loader)
            public_test_loader, private_test_loader = dataset_shuffle(public_test_loader, private_test_loader)
            test_comms_total.append(test_comms)

        print("\nTraining statistics: ")
        print("Total training time: ", tot_train_time)
        if args.comm_info:
            comm_sum = np.sum(comms_total) + model_comms
            print(f"Total communication overhead for training: {comm_sum} bytes")
            print(f"Total client communication overhead: {client_server_comms} bytes")
            print(f"Total server communication overhead: {server_client_comms} bytes")

        if args.comm_info:
            print("\nTesting statistics: ")
            test_comm_sum = np.sum(test_comms_total)
            print(f"Total communication overhead for testing: {test_comm_sum} bytes")




    else:
        test_time = test(args, model, private_test_loader)
        if not args.test:
            print(
                f"{ 'Online' if args.preprocess else 'Total' } time (s):\t",
                round(test_time / args.batch_size, 4),
            )
        else:
            # Compare with clear text accuracy
            print("Clear text accuracy is:")
            model = get_model(
                args.model, args.dataset, out_features=get_number_classes(args.dataset)
            )
            load_state_dict(model, args.model, args.dataset)
            test(args, model, public_test_loader)

    if args.preprocess:
        missing_items = [len(v) for k, v in sy.preprocessed_material.items()]
        if sum(missing_items) > 0:
            print("MISSING preprocessed material")
            for key, value in sy.preprocessed_material.items():
                print(f"'{key}':", value, ",")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        help="model to use for inference (split, full)",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        help="dataset to use (mnist, cifar10, hymenoptera, tiny-imagenet)",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        help="size of the batch to use. Default 128.",
        default=128,
    )
    parser.add_argument(
        "--test_batch_size",
        type=int,
        help="size of the batch to use",
        default=None,
    )

    parser.add_argument(
        "--preprocess",
        help="[only for speed test] preprocess data or not",
        action="store_true",
    )

    parser.add_argument(
        "--fp_only",
        help="Don't secret share values, just convert them to fix precision",
        action="store_true",
    )

    parser.add_argument(
        "--public",
        help="[needs --train] Train without fix precision or secret sharing",
        action="store_true",
    )

    parser.add_argument(
        "--test",
        help="run testing on the complete test dataset",
        action="store_true",
    )

    parser.add_argument(
        "--train",
        help="run training for n epochs",
        action="store_true",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        help="[needs --train] number of epochs to train on. Default 15.",
        default=15,
    )

    parser.add_argument(
        "--lr",
        type=float,
        help="[needs --train] learning rate of the SGD. Default 0.01.",
        default=0.01,
    )

    parser.add_argument(
        "--momentum",
        type=float,
        help="[needs --train] momentum of the SGD. Default 0.9.",
        default=0.9,
    )

    parser.add_argument(
        "--verbose",
        help="show extra information and metrics",
        action="store_true",
    )

    parser.add_argument(
        "--log_interval",
        type=int,
        help="[needs --test or --train] log intermediate metrics every n batches. Default 10.",
        default=10,
    )

    parser.add_argument(
        "--comm_info",
        help="Print communication information",
        action="store_true",
    )


    cmd_args = parser.parse_args()

    # Sanity checks

    if cmd_args.test or cmd_args.train:
        assert (
            not cmd_args.preprocess
        ), "Can't preprocess for a full epoch evaluation or training, remove --preprocess"

    if cmd_args.train:
        assert not cmd_args.test, "Can't set --test if you already have --train"

    if cmd_args.fp_only:
        assert not cmd_args.preprocess, "Can't have --preprocess in a fixed precision setting"
        assert not cmd_args.public, "Can't have simultaneously --fp_only and --public"

    if not cmd_args.train:
        assert not cmd_args.public, "--public is used only for training"



    class Arguments:
        model = cmd_args.model.lower()
        dataset = cmd_args.dataset.lower()
        preprocess = cmd_args.preprocess        
        verbose = cmd_args.verbose

        train = cmd_args.train
        n_train_items = -1 if cmd_args.train else cmd_args.batch_size
        test = cmd_args.test or cmd_args.train
        n_test_items = -1 if cmd_args.test or cmd_args.train else cmd_args.batch_size

        batch_size = cmd_args.batch_size
        # Defaults to the train batch_size
        test_batch_size = cmd_args.test_batch_size or cmd_args.batch_size

        log_interval = cmd_args.log_interval
        comm_info = cmd_args.comm_info

        epochs = cmd_args.epochs
        lr = cmd_args.lr
        momentum = cmd_args.momentum

        public = cmd_args.public
        fp_only = cmd_args.fp_only
        requires_grad = cmd_args.train
        dtype = "long"
        protocol = "fss"
        precision_fractional = 5 if cmd_args.train else 4

    args = Arguments()

    run(args)
