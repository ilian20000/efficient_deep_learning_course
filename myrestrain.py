import argparse
import matplotlib.pyplot as plt

import resnet
import densenet
from mytrainer import NetTrainer

parser = argparse.ArgumentParser(prog="Training script for resnet optimization",
                                 description="Automatically rains the python net given in the first argument",
                                 epilog="the end")

parser.add_argument('-d', '--debug', action='store_true',
                    help='Speedruns the training process and benchmark to ensure there is no obvious error')
parser.add_argument('-s', '--save', action='store_true',
                    help='Saves a copy of the net every epoch')

args = parser.parse_args()


if __name__ == "__main__":
    g_nepochs = 300
    g_neval = 10

    mynet1 = densenet.densenet_cifar()
    trainer1 = NetTrainer(mynet1, args)
    # trainer1.load("saves/finaltrain_resnet")
    # trainer1.to_half()
    trainer1.nepochs = g_nepochs
    trainer1.trainloop(n_eval=g_neval)
    trainer1.benchmark()
    trainer1.save("saves/densecifarLRSCHEDULE300epochs")
    plt.subplot(1, 2, 1)
    plt.plot(trainer1.benchstats['epoch'], trainer1.benchstats['test accuracy'],  c='blue', label='net1 test accuracy')
    plt.plot(trainer1.benchstats['epoch'], trainer1.benchstats['train accuracy'], '--', c='blue',  label='net1 train accuracy')
    plt.xlabel("epoch")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(trainer1.benchstats['epoch'], trainer1.benchstats['train loss'],  c='green', label='net1 loss')
    
    plt.savefig("saves/densecifar1Schelduled.png")

# Hyperparameters
# Data augmentation
# Quantization
# Pruning

