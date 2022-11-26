import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

def getRandomGrid(size : int):
    """ Return a random 2D grid with values between 1 to 255 of input size """
    return np.random.randint(1, 255, (size, size, 3)).astype(np.float32)


def getRandomTrainData(size : int):
    """ Return a random 2D grid with values between 1 to 255 of input size """
    return np.random.randint(1, 255, (size, 3)).astype(np.float32)


def visualizeGridImg(images : np.ndarray):
    # Grid will always be square, get the size of it
    size = images.shape[0]

    # Loop over image grid and concatenate it to a one single image which then is visualized
    toVis = None
    for i in range(size):
        for j in range(size):
            img = images[i][j].data
            if i == 0 and j == 0:
                toVis = img
            else:
                toVis = np.concatenate((toVis, img), axis=(i, j))
    cv2.imshow("Grid", toVis)


def gallery(images):
    """ Visualize a 2D grid of images """
    # Get the size of the array
    numImgs = images.shape[0]
    width, height, channels = images[0][0].data.shape

    # Initialize an empty canvas
    canvas = np.zeros((numImgs * width, numImgs * height, channels), dtype=np.uint8)

    # Loop over the array and put each image on the canvas
    for i in range(numImgs):
        for j in range(numImgs):
            canvas[i * width: (i + 1) * width, j * height: (j + 1) * height] = images[i][j].data
    return canvas
    

def resizeImg(img, percentage):
    """ Resizes an image according to a percentage """
    #calculate the new widht and height
    width = int(img.shape[1] * percentage / 100)
    height = int(img.shape[0] * percentage / 100)

    # dsize
    size = (width, height)

    # resize image
    return cv2.resize(img, size)


def visualizeGrid(grid : np.ndarray):
    grid = grid.astype(np.uint8)
    cv2.imshow("Grid", grid)
    cv2.waitKey(0)


def saveImg(img : np.ndarray, epoch : str):
    outputPath = os.path.join("results", "epoch_" + epoch + ".png")
    cv2.imwrite(outputPath, img)


def visTwoImages(img1 : np.ndarray, img2 : np.ndarray):
    img1 = img1.astype(np.uint8)
    img2 = img2.astype(np.uint8)

    img1 = np.pad(img1, ((1,1), (1,1), (0, 0)), mode='constant', constant_values=0)
    img2 = np.pad(img2, ((1,1), (1,1), (0, 0)), mode='constant', constant_values=0)
    toVis = np.concatenate((img1, img2), axis=1)

    cv2.imshow("Grid", toVis)
    cv2.waitKey(0)


def plotXY(input):
    x = [entry[0] for entry in input]
    y = [entry[1] for entry in input]
    plt.plot(x, y, 'ro')
    plt.show()


def loadImagesFrom(path, everyNth=0):
    """ Load images from a given path with a possibility to skip every Nth image for smaller sample sizes """
    if(os.path.exists(path)):
        # Loop over all of the images in the path and save them to a numpy list
        images = []
        counter = 1
        for dir, subdir, filenames in os.walk(path):
            # Continue if theres no files
            if len(filenames) < 1:
                continue
            
            for filename in filenames:
                counter += 1
                if (filename.endswith(".png") or filename.endswith(".jpg")) and counter % everyNth == 0:
                    img = cv2.imread(os.path.join(dir, filename))
                    img = resizeImg(img, 25)
                    images.append(img)
        return np.asarray(images)


def sampleListRandomly(input : np.ndarray, n=10):
    """ Returns a list of a randomly sampled input list """
    return np.random.choice(input, size=(n, n))


def computeHogFeatures(image : np.ndarray):
    """ Returns HOG image feature descriptor """
    cellSize = (16, 16)  # h x w in pixels
    blockSize = (2, 2)  # h x w in cells
    bins = 18  # number of orientation bins
    hog = cv2.HOGDescriptor(_winSize=(image.shape[0] // cellSize[0] * cellSize[0],
                                      image.shape[1] // cellSize[1] * cellSize[1]),
                            _blockSize=(blockSize[0] * cellSize[0],
                                        blockSize[1] * cellSize[1]),
                            _blockStride=(cellSize[0], cellSize[1]),
                            _cellSize=(cellSize[0], cellSize[1]),
                            _signedGradient=True,
                            _nbins=bins)

    return hog.compute(image.astype(np.uint8))


def getEuclideanDistance(img1 : np.ndarray, img2 : np.ndarray):
    """ Calculates the euclidean distance between two HOG feature vectors """
    return np.linalg.norm(img1 - img2)