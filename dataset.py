import os
from torchvision.datasets import CIFAR100
dataset = CIFAR100(root="./dataset",download=True,train=False)
# testdat
if __name__ == "__main__":
    imgs,labels = dataset[100]
    print(imgs,type(imgs))
    print(labels,type(labels))
    print(len(dataset))
