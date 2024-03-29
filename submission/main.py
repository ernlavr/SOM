#!/usr/bin/env python3

import argparse
import os

import cv2
import numpy as np

import src.SOM as som
from src.Utils import *


# Create commandline arguments for passing location of image
def getCmdLineArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default="", help='Input data')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose')

    args = parser.parse_args()
    return args


def main():
    args = getCmdLineArgs()
    inputData = loadImagesFrom(args.input, everyNth=5)
    verbose = args.verbose
    map = som.SOM(inputData, verbose)
    

if __name__ == "__main__":
    main()
