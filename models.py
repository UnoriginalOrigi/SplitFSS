import os
import re

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

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
    
class ConvLayers(nn.Module):
    def __init__(self, dataset):
        super(ConvLayers, self).__init__()
        if dataset == "cifar10":
            self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=0)
        else:
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

#SecureNN esque models
class Network1(nn.Module): #server regular split model
    def __init__(self, dataset, out_features):
        super(Network1, self).__init__()
        self.fc_layers = FCLayers(out_features)

    def forward(self, x):
        x = self.fc_layers(x)
        return x

class Network2(nn.Module): #client model
    def __init__(self, dataset, out_features):
        super(Network2, self).__init__()
        self.conv_layers = ConvLayers(dataset)

    def forward(self, x):
        x = self.conv_layers(x)
        return x
    
class Network3(nn.Module): #server usplit model
    def __init__(self, dataset, out_features):
        super(Network3, self).__init__()
        self.fc1 = nn.Linear(256, 100)
        self.fc2 = nn.Linear(100, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x
    
class FullModel(nn.Module):
    def __init__(self, dataset, out_features):
        super(FullModel, self).__init__()
        self.conv_layers = ConvLayers(dataset)
        self.fc_layers = FCLayers(out_features)

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

#Original LeNet models
class FullLeNet(nn.Module):
    def __init__(self, dataset, out_features=10):
        super(FullLeNet, self).__init__()
        if dataset == "cifar10":
            self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=0)
        else:
            self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)
        self.fc1 = nn.Linear(400, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, out_features)

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(x)  
        x = self.conv2(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(x)  
        x = x.reshape(-1, 400)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        return x
    
class SplitLeNet1(nn.Module):
    def __init__(self, dataset, out_features=10):
        super(SplitLeNet1, self).__init__()
        if dataset == "cifar10":
            self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=0)
        else:
            self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(x)  
        x = self.conv2(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(x)  
        x = x.reshape(-1, 400)
        return x
    
class SplitLeNet2(nn.Module):
    def __init__(self, dataset, out_features=10):
        super(SplitLeNet2, self).__init__()
        self.fc1 = nn.Linear(400, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        return x
    
class SplitLeNet3(nn.Module):
    def __init__(self, dataset, out_features=10):
        super(SplitLeNet3, self).__init__()
        self.fc1 = nn.Linear(400, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x

model_zoo = {
    "split_priv": Network1,
    "split_pub": Network2,
    "usplit": Network3,
    "full": FullModel,
    "lefull": FullLeNet,
    "lesplit_pub": SplitLeNet1,
    "lesplit_priv": SplitLeNet2,
    "leusplit": SplitLeNet3
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
