import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import torch.utils.data as DataUtils

DATA_ROOT='./MNISTData/'
def getMNISTDataLoaders(batchSize=64, nTrain=50000, nVal=10000, nTest=10000):
    # You can use technically use the same transform instance for all 3 sets
    assert (60000 - nVal) == nTrain, 'nTrain + nVal must be equal to 60000'
    trainTransform = transforms.Compose([transforms.ToTensor()])
    valTransform = transforms.Compose([transforms.ToTensor()])
    testTransform = transforms.Compose([transforms.ToTensor()])

    trainSet = datasets.MNIST(root=DATA_ROOT, download=True, train=True, \
                            transform=trainTransform)
    valSet = datasets.MNIST(root=DATA_ROOT, download=True, train=True, \
                            transform=valTransform)
    testSet = datasets.MNIST(root=DATA_ROOT, download=True, train=False, \
                                    transform=testTransform)

    indices = np.arange(0, 60000)
    np.random.shuffle(indices)

    trainSampler = SubsetRandomSampler(indices[:nTrain])
    valSampler = SubsetRandomSampler(indices[nTrain:])
    testSampler = SubsetRandomSampler(np.arange(0, nTest))

    trainLoader = DataUtils.DataLoader(trainSet, batch_size=batchSize, \
                                    sampler=trainSampler)
    valLoader = DataUtils.DataLoader(valSet, batch_size=batchSize, \
                                    sampler=valSampler)
    testLoader = DataUtils.DataLoader(testSet, batch_size=batchSize, \
                                    sampler=testSampler)
    return trainLoader, valLoader, testLoader