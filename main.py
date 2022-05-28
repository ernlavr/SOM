import argparse
import os

import cv2
import numpy as np

import src.SOM as som


# Create commandline arguments for passing location of image
def getCmdLineArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default="", help='Input data')
    args = parser.parse_args()
    return args


def main():
    #args = getCmdLineArgs()
    # img = cv2.imread(args.input)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    map = som.SOM(None)
    

if __name__ == "__main__":
    main()
