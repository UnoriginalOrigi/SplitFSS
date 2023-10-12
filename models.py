import os
import re

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class Network1(nn.Module): #server model
    def __init__(self, dataset, out_features):
        super(Network1, self).__init__()
        self.fc_layers = FCLayers(out_features)

    def forward(self, x):
        x = self.fc_layers(x)
        return x


class ConvLayers(nn.Module):
    def __init__(self):
        super(ConvLayers, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=0)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=5, stride=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(x)
        x = x.reshape(-1, 256)
        return x
    
class Network3(nn.Module):
    def __init__(self, dataset, out_features):
        super(Network3, self).__init__()
        self.fc1 = nn.Linear(256, 100)
        self.fc2 = nn.Linear(100, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


class FCLayers(nn.Module):
    def __init__(self, out_features):
        super(FCLayers, self).__init__()
        self.fc1 = nn.Linear(256, 100)
        self.fc2 = nn.Linear(100, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        return x

class Network2(nn.Module): #client model
    def __init__(self, dataset, out_features):
        super(Network2, self).__init__()
        self.conv_layers = ConvLayers()

    def forward(self, x):
        x = self.conv_layers(x)

        return x
    
class FullModel(nn.Module): #client model
    def __init__(self, dataset, out_features):
        super(FullModel, self).__init__()
        self.conv_layers = ConvLayers()
        self.fc_layers = FCLayers(out_features)

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x



model_zoo = {
    "split_priv": Network1,
    "split_pub": Network2,
    "usplit": Network3,
    "full": FullModel,
}


def get_model(model_name, dataset, out_features):
    return model_zoo[model_name](dataset, out_features)


online_models = {
    "lenet_mnist": {
        "id": "1WWh_POWmgcBEDxk87t50DEZmTik9NRkg",
        "file_name": "lenet_mnist_baseline_99.27.pt",
    },
    "alexnet_cifar10": {
        "id": "1-M8SaF19EFSI1Zqmnr9KL5aQG2AEqWND",
        "file_name": "alexnet_cifar10_baseline_70.23.pt",
    },
    "alexnet_tiny-imagenet": {
        "id": "1Nygb3K8dbSBYMls3U6rngYIAYrRsLwR0",
        "file_name": "alexnet_tiny-imagenet_baseline_37.8.pt",
    },
    "resnet18_hymenoptera": {
        "id": "1bNHE91Fn32AGPNyk_hmGZuQdpnVmyOtR",
        "file_name": "resnet18_hymenoptera_95.pt",
    },
}

too_big_models = {
    "vgg16_cifar10": "17k1nKItmp-4E1r5GFqfs8oH1Uhmp5e_0",
    "vgg16_tiny-imagenet": "1uBiLpPi34Z3NywW3zwilMZpmb964oU8q",
}


def load_state_dict(model, model_name, dataset):
    MODEL_PATH = "pretrained_models/"

    base_name = f"{model_name}_{dataset}"
    file_name = None
    for file in os.scandir(MODEL_PATH):
        if re.match(fr"^{base_name}", file.name):
            file_name = file.name

    if file_name is None:
        if base_name in online_models:
            id = online_models[base_name]["id"]
            file_name = online_models[base_name]["file_name"]
            print(f"Downloading model {file_name}... ")
            os.system(
                f"wget --no-check-certificate "
                f"'https://docs.google.com/uc?export=download&id={id}' -O {MODEL_PATH+file_name}"
            )
        else:
            if base_name in too_big_models:
                id = too_big_models[base_name]
                print(
                    f"Model {base_name} has to be downloaded manually :( \n\n"
                    f"https://docs.google.com/uc?export=download&id={id}\n"
                )
            raise FileNotFoundError(f"No pretrained model for {model_name} {dataset} was found!")
    model.load_state_dict(torch.load(MODEL_PATH + file_name, map_location=torch.device("cpu")))
    print(f"Pre-trained model loaded from {file_name}")
