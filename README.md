# SplitFSS


### Info

This is the open-source implementation to reproduce the results of Split Learning combined with Function Secret Sharing

This code is based on the original [AriaNN framework paper](https://arxiv.org/abs/2006.04593) and [AriaNN code](https://github.com/LaRiffle/ariann/).


### Models
This implementation provides two different models with three different variations each. The core models are:
* MiniONN consisting of:
  * Convolution Layer (16 Output Channels, Kernel Size: 5, Stride: 1)
  * Max Pool (Kernel: 2, Stride: 2) + ReLU
  * Convolution (16 Output Channels, Kernel Size: 5, Stride: 1)
  * Max Pool (Kernel: 2, Stride: 2) + ReLU
  * Fully Connected Layer (256 inputs, 100 outputs) + ReLU
  * Fully Connected Layer (100 inputs, dataset_amount outputs) + ReLU
* LeNet consisting of:
  * Convolution Layer (6 Output Channels, Kernel Size: 5, Stride: 1)
  * Max Pool (Kernel: 2, Stride: 2) + ReLU
  * Convolution (16 Output Channels, Kernel Size: 5, Stride: 1)
  * Max Pool (Kernel: 2, Stride: 2) + ReLU
  * Fully Connected Layer (400 inputs, 120 outputs) + ReLU
  * Fully Connected Layer (120 inputs, 84 outputs) + ReLU
  * Fully Connected Layer (84 inputs, dataset_amount outputs) + ReLU

Each model has three variations:
* Full model -- consisting of no use of Split Learning, the full model.
* Split model -- model architectures are split after the second Convolution + Max Pool layer, where the server does both Fully Conneted Layers and starts the backpropogation after receiving the correspondings labels.
* USplit model -- same as the Split model, but the final layer is executed back on the client side and the client does not have to share the labels.

### Datasets
The implementation has 3 different datasets:
* MNIST -- 60 000 images (50 000 training, 10 000 testing) 28x28x1 size, which contains monochrome images of handwritten numbers. A standard machine learning dataset for testing.
* FashionMNIST -- 60 000 images (50 000 training, 10 000 testing) 28x28x1 size, which contains images of standard clothing items. More difficult and robust than the original MNIST dataset.
* CIFAR-10 -- 60 000 images (50 000 training, 10 000 testing) 28x28x3 size, which contains colored images of diverse real world items in centre frame. More channels and more difficult to correctly classify.
## Usage

### Reproduce results

You may use the premade experiment files in `/experiments` to reproduce the results of all of our tested models.

### Personalized tests
To train different models or set different parameters you can set each parameter following the documentation. An example of running the private vanilla SL model for less epochs and a different learning rate and momentum:
```
python3 main.py --model split --dataset mnist --train --epochs 5 --lr 0.01 --momentum 0.95 --comm_info
```
Adding the parameter ``` --public ``` to the command will perform standard training without the use of Function Secret Sharing.

### Documentation

```
usage: main.py [-h] [--model MODEL] [--dataset DATASET] [--batch_size BATCH_SIZE] [--test_batch_size TEST_BATCH_SIZE] [--fp_only] [--public] [--train] [--epochs EPOCHS]
               [--lr LR] [--momentum MOMENTUM] [--verbose] [--log_interval LOG_INTERVAL] [--comm_info]

optional arguments:
  -h, --help            show this help message and exit
  --model MODEL         model to use (full, split, usplit, lefull, lesplit, leusplit)
  --dataset DATASET     dataset to use (mnist, fmnist, cifar10)
  --batch_size BATCH_SIZE
                        size of the batch to use. Default 128.
  --test_batch_size TEST_BATCH_SIZE
                        size of the batch to use
  --fp_only             Don't secret share values, just convert them to fix precision
  --public              [needs --train] Train without fix precision or secret sharing
  --train               run training for n epochs
  --epochs EPOCHS       [needs --train] number of epochs to train on. Default 15.
  --lr LR               [needs --train] learning rate. Default 0.01.
  --momentum MOMENTUM   [needs --train] momentum. Default 0.9.
  --verbose             show extra information and metrics
  --log_interval LOG_INTERVAL
                        [needs --test or --train] log intermediate metrics every n batches. Default 10.
  --comm_info           Print communication information
```

## Installation

We recommend using a fresh Python 3.7 environment, when running the experiments.

#### PySyft

Download PySyft from GitHub using the `ryffel/ariaNN` branch and install in editable mode:
```
git clone https://github.com/OpenMined/PySyft.git
cd PySyft
git checkout a73b13aa84a8a9ad0923d87ff1b6c8c2facdeaa6
pip install -e .
```

This should allow you to run the experiments for reproducing our results.
 
#### Troubleshooting

* The needed checkout in PySyft has errors when running the installation process and throws a `error: metadata-generation-failed`. In this case you need to access the PySyft directory and find `.../PySyft/pip-dep/requirements_udacity.txt` and change the `tf_encrypted` line to `tf_encrypted>=0.5.4`.
* On Linux make sure all needed libraries are installed acording to your distribution such as `libsrtp2-dev`, `libavformat-dev` and `libavdevice-dev`.
* If there are errors when running the code for a missing module `ModuleNotFoundError: No module named 'torchcsprng'`, then install it through pip with `pip3 install torchcsprng` and then downgrading manually your `torch` version to 1.4.0 by running `pip install torch==1.4.0`.
* In the case that `protobuf` throws an error when running the code, downgrade the version of it to 3.20.3 by running `pip install protobuf==3.20.3`.
* We used a CPU for our tests, as such we disable CUDA capable devices by running `export CUDA_VISIBLE_DEVICES==""` before running the code.
