import csv
import numpy
import sklearn
from datetime import *
reader = csv.reader(open("Occupancy_Estimation.csv", "r"),
                    delimiter=",")
x = list(reader)
