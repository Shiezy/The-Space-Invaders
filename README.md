# Space-Invaders
An implementation of deep learning on Space invaders

#Preliquisites
install all the following modules using pip3 on python>3.3

import tensorflow as tf      # Deep Learning library
import numpy as np           # Handle matrices
import retro                 # Retro Environment


from skimage import transform # Help us to preprocess the frames
from skimage.color import rgb2gray # Help us to gray our frames

import matplotlib.pyplot as plt # Display graphs

from collections import deque# Ordered collection with ends

import random


#Cannot find the game
Download the space invaders rom and import it using retro
