import numpy as np
import random
import cv2
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from src.Structs import *
from src.Utils import *

class SOM():

    def __init__(self, input, verbose):
        # SOM variables
        """ Training epochs"""
        self.epochs = 15
        """ X Coordinate of the BMU"""
        self.xBMU = 0
        """ Y Coordinate of the BMU"""
        self.yBMU = 0
        """ Learning rate"""
        self.learningRate = 1
        """ Decay rate of the learning rate"""
        self.alpha = 0.01
        """ Size of the SOM """
        self.mapSize = 256
        """ Radius"""
        self.radius = self.mapSize / 4
        """ Radius decay rate """
        self.beta = 0.01

        # Training data and network, initialized in the next stesp
        self.trainingData = None
        self.SOM = None

        # Debug
        self.learningRates = []
        self.radiuses = []
        self.verbose = verbose

        if input is not None:
            self.trainingData = self.parseImageDataset(input)
            self.SOM = sampleListRandomly(self.trainingData, 20)
        else:
            self.trainingData = getRandomTrainData(1000)
            self.SOM = getRandomGrid(self.mapSize)

        # Visualize the grid
        # Initialize the training data
        original = self.SOM.copy()
        newSom = self.trainSOM(self.SOM, self.trainingData)
        
        
        visualizeGrid(gallery(newSom))
        if self.verbose is True:
            pass
            #plotXY(self.learningRates)
            #plotXY(self.radiuses)
        

    def parseImageDataset(self, input):
        """ Stores all images and their feature vectors in a data class """
        tmp = np.array([], dtype=Image)
        for i, img in enumerate(input):
            features = computeHogFeatures(img)
            tmp = np.append(tmp, Image(img, features))
            if self.verbose is True:
                print(f"Processing image: {i} / {len(input)}")
        
        return tmp
            


    def findBmu(self, SOM, x):
        
        # compare euclidean distance of each element of the SOM to the input
        # and save the index of the closest element to bmu
        distances = np.zeros((SOM.shape[0], SOM.shape[1]), dtype=np.float32)
        for width in range(len(SOM)):
            for length in range(len(SOM)):
                element = SOM[width][length].features
                # Get euclidean distance from two float arrays element and x.features
                d = np.linalg.norm(element - x.features)
                # Insert the euclidean distance into the distances array
                distances[width][length] = d
                

        # Get the index of the minimum distance
        min_index = np.unravel_index(np.argmin(distances, axis=None), distances.shape)
        # return the coordinates of the minimum distance
        return min_index
        


    def getUpdatedLearnRate(self, learningRate, epoch):
        """ Update learning rate by an exponent """
        return learningRate * np.exp(epoch * -self.alpha)


    def getUpdatedRadius(self, radius, epoch):
        """ Gradually shrink the radius """
        return radius * np.exp(epoch * -self.beta)


    def updateWeights(self, SOM, trainingExample, learningRate, radius, bmuCoord, step=2):
        g, h = bmuCoord
        
        # Change cells in a neighbourhood of BMU
        for i in range(max(0, g - step), min(SOM.shape[0], g + step)):
            for j in range(max(0, h - step), min(SOM.shape[1], h + step)):
                # distSq = np.square((i - g) ** 2) + np.square((j - h) ** 2)
                # distFunc = np.exp(-distSq / 2 / radius)


                # img = learningRate * distFunc * (trainingExample.data - SOM[i, j].data)
                # features = computeHogFeatures(img)
                # toAdd = Image(img, features)
                SOM[i, j] = trainingExample
                return SOM


    def trainSOM(self, SOM, trainingData):
        for epoch in range(self.epochs):
            print(f"Epoch: {epoch} / {self.epochs}")
            np.random.shuffle(trainingData)
            gridImg = gallery(SOM)
            saveImg(gridImg, str(epoch))
            for trainingExample in trainingData:
                g, h = self.findBmu(SOM, trainingExample)
                SOM = self.updateWeights(SOM, trainingExample, self.learningRate, self.radius, (g, h))
            
            # Update the learning rate and radius
            self.learningRate = self.getUpdatedLearnRate(self.learningRate, epoch)
            self.radius = self.getUpdatedRadius(self.radius, epoch)

            radiusEntry = (epoch, self.radius)
            learningRateEntry = (epoch, self.learningRate)

            if self.verbose is True:
                self.radiuses.append(radiusEntry)
                self.learningRates.append(learningRateEntry)
                print(f"Radius: {self.radius}; Learning rate: {self.learningRate}")

        return SOM



    def neighbourhoodDistanceFunction(self, x, y, r):
        """ Calculate the distance between two points"""
        pass

    def train(self, input, epochs, radius, learning_rate):
        """ Implement the training function of a self organizing map"""
        pass