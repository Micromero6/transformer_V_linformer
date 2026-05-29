import math
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tabulate import tabulate
from transformers import DisilBertTokenizer
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidVectorizer
from sklearn.metrics import accuracy_score