# SplitFSS


### Info

This is the open-source implementation to reproduce the results of Split Learning combined with Function Secret Sharing

This code is based on the original [AriaNN framework paper](https://arxiv.org/abs/2006.04593) and [AriaNN code](https://github.com/LaRiffle/ariann/).



## Usage

### Reproduce results

You may use the premade experiment files in `/experiments` to reproduce the results of all of our tested models.

### Personalized tests
To train different models or set different parameters you can set each parameter following the documentation. An example of running the private vanilla SL model for less epochs and a different learning rate and momentum:
```
python3 main.py --model split --dataset mnist --train --epochs 5 --lr 0.01 --momentum 0.95 --comm_info
```

### Documentation

```
usage: main.py [-h] [--model MODEL] [--dataset DATASET] [--batch_size BATCH_SIZE] [--test_batch_size TEST_BATCH_SIZE] [--fp_only] [--public] [--train] [--epochs EPOCHS]
               [--lr LR] [--momentum MOMENTUM] [--verbose] [--log_interval LOG_INTERVAL] [--comm_info]

optional arguments:
  -h, --help            show this help message and exit
  --model MODEL         model to use (full, split, usplit)
  --dataset DATASET     currently only MNIST dataset is available (mnist)
  --batch_size BATCH_SIZE
                        size of the batch to use. Default 128.
  --test_batch_size TEST_BATCH_SIZE
                        size of the batch to use
  --fp_only             Don't secret share values, just convert them to fix precision
  --public              [needs --train] Train without fix precision or secret sharing
  --train               run training for n epochs
  --epochs EPOCHS       [needs --train] number of epochs to train on. Default 15.
  --lr LR               [needs --train] learning rate of the SGD. Default 0.01.
  --momentum MOMENTUM   [needs --train] momentum of the SGD. Default 0.9.
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
cd ~
git clone https://github.com/OpenMined/PySyft.git
cd PySyft
git checkout a73b13aa84a8a9ad0923d87ff1b6c8c2facdeaa6
pip install -e .
```

This should allow you to run the experiments for reproducing our results.
