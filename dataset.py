import os
import clip
from torchvision.datasets import CIFAR100
dataset = CIFAR100(root="./dataset",download=True,train=False)
model,preprocess = clip.load("RN50")
classes = dataset.classes
# testdat
if __name__ == "__main__":
    imgs,labels = dataset[100]
    print(imgs,type(imgs))
    print(preprocess(imgs).shape)
    print(labels,type(labels))
    print(len(dataset))
    print(dataset.classes)
