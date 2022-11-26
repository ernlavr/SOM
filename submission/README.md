# SOM
Self Organizing Map for clustering images based on their HOG features


### Prerequisites
- Developed with Python 3.10.4
- Dependencies: Numpy, OpenCV, Matplotlib


### To Run
Make sure you have MiniConda installed. Otherwise install dependencies as per prerequisites and skip Step 1.
1. Create the conda environment `conda env create -f environment.yml`
2. Run `main.py` and use `-i` flag to point to an image set folder. This will recursively scan all `.png` and `.jpg` images in the presented folder and its sub-folders \
i.e. `$ python3 main.py -i data/afhq/val/`
3. Observe the output, each epoch is also written in `results/` folder

Dataset from https://www.kaggle.com/andrewmvd/animal-faces