import os

import torch
from torchvision import datasets, transforms

HOME = "~"


def get_number_classes(dataset):
    number_classes = {
        "mnist": 10,
        "cifar10": 10,
    }
    return number_classes[dataset]


def one_hot_of(index_tensor):
    """
    Transform to one hot tensor

    Example:
        [0, 3, 9]
        =>
        [[1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]]

    """
    onehot_tensor = torch.zeros(*index_tensor.shape, 10)  # 10 classes for MNIST
    onehot_tensor = onehot_tensor.scatter(1, index_tensor.view(-1, 1), 1)
    return onehot_tensor


def get_data_loaders(args, kwargs, private=True):
    def encode(tensor):
        """
        Depending on the setting, acts on a tensor
        - Do nothing
        OR
        - Transform to fixed precision
        OR
        - Secret share
        """
        if args.public:
            return tensor

        encrypted_tensor = tensor.encrypt(**kwargs)
        if args.fp_only: 
            return encrypted_tensor.get()
        return encrypted_tensor

    dataset = args.dataset

    if dataset == "mnist":
        if args.model == "lefull" or args.model == "lesplit":
            transformation = transforms.Compose(
                [transforms.Resize((32,32)), transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            )
        else:
            transformation = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            )
        train_dataset = datasets.MNIST(
            HOME+"/data", train=True, download=True, transform=transformation
        )
        test_dataset = datasets.MNIST(
            HOME+"/data", train=False, download=True, transform=transformation
        )
    elif dataset == "cifar10":
        transformation = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        train_dataset = datasets.CIFAR10(
            HOME+"/data", train=True, download=True, transform=transformation
        )
        test_dataset = datasets.CIFAR10(
            HOME+"/data", train=False, download=True, transform=transformation
        )
    else:
        raise ValueError(f"Not supported dataset {dataset}")
    


    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle = True,
        drop_last=True,
    )

    new_train_loader_pub = []
    new_train_loader_priv = []
    for i, (data, target) in enumerate(train_loader):
        if args.n_train_items >= 0 and i >= args.n_train_items / args.batch_size:
            break
        
        new_train_loader_priv.append((encode(data), encode(one_hot_of(target))))
        new_train_loader_pub.append((data, one_hot_of(target)))

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        shuffle = True,
        drop_last=True,
    )

    new_test_loader_pub = []
    new_test_loader_priv = []
    for i, (data, target) in enumerate(test_loader):
        if args.n_test_items >= 0 and i >= args.n_test_items / args.test_batch_size:
            break

        new_test_loader_priv.append((encode(data), encode(target.float())))
        new_test_loader_pub.append((data, target))

    return new_train_loader_pub, new_train_loader_priv, new_test_loader_pub, new_test_loader_priv
