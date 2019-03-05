import numpy as np
import matplotlib.pyplot as plt
import time


class kMeansAlgo:

    def __init__(self):
        pass

    def fit(self, data, k, threshold):
        self.data = data
        self.k = k
        self.threshold = threshold
        self.initializeCentroids()
        self.startAlgo()

    def initializeCentroids(self):
        self.noOfDataPoints = self.data.shape[0]
        self.centroidsArr = self.data[np.random.randint(
            low=0, high=self.noOfDataPoints - 1, size=self.k)]

    def startAlgo(self):
        distanceFromSeedsMatrix = np.zeros([self.noOfDataPoints, k])
        absChange = 100
        prevDistSum = -100
        count = 0
        while absChange >= self.threshold:
            for i in range(k):
                centroidRepeatArr = np.array(
                    [self.centroidsArr[i], ] * self.noOfDataPoints)
                distanceFromSeedsMatrix[:, i] = np.sum(
                    (self.data - centroidRepeatArr)**2, axis=1)  # sum along row
            self.algoAssignedLabels = np.argmin(
                distanceFromSeedsMatrix, axis=1)
            nextDistSum = np.sum(distanceFromSeedsMatrix)
            for i in range(k):
                idx = (self.algoAssignedLabels == i)
                dataPtsWithLabelI = self.data[idx]
                self.centroidsArr[i, :] = np.sum(
                    dataPtsWithLabelI, axis=0) / dataPtsWithLabelI.shape[0]    # sum along column
            absChange = abs(nextDistSum - prevDistSum)
            prevDistSum = nextDistSum
            count += 1
        self.plotClustersAndCentroids()

    def plotClustersAndCentroids(self):
        for i in range(k):
            xFirstCoordinate = [self.data[t, 0] for t in range(
                len(self.algoAssignedLabels)) if self.algoAssignedLabels[t] == i]
            xSecondCoordinate = [self.data[t, 1] for t in range(
                len(self.algoAssignedLabels)) if self.algoAssignedLabels[t] == i]
            plt.scatter(xFirstCoordinate, xSecondCoordinate)
        centroidsFirstCoordinates = self.centroidsArr[:, 0]
        centroidsSecondCoordinates = self.centroidsArr[:, 1]
        plt.scatter(centroidsFirstCoordinates,
                    centroidsSecondCoordinates, marker="X")
        plt.show()


start_time = time.time()
kMeansAlgoObj = kMeansAlgo()

data = np.concatenate((np.random.normal(loc=-20, scale=3, size=[40000, 2]), np.random.normal(
    loc=0, scale=3, size=[40000, 2]), np.random.normal(loc=20, scale=3, size=[40000, 2])))
k = 3
threshold = 1
kMeansAlgoObj.fit(data, k, threshold)
print("--- %s seconds ---" % (time.time() - start_time))
