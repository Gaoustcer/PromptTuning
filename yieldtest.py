def yieldfunc():
    while 1:
        for i in range(3):
            yield i
def yielddataset():
    from dataset import dataset
    print("len of dataset",len(dataset))
    # exit()
    # i = 0
    while 1:
        for img,label in dataset:
            yield img,label
            # i += 1
def yieldanother():
    from dataset import dataset
    while 1:
        for img,label in dataset:
            yield label


if __name__ == "__main__":
    for num in yielddataset():
        print(num)  
        