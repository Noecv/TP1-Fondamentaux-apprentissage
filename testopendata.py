import csv
import numpy
from datetime import datetime
reader = csv.reader(open("Occupancy_Estimation.csv", "r"),
                    delimiter=",")
x = list(reader)
print(type(x[1][0]))

# result = numpy.array(x).astype("float")

x = x[1:]


def str_date_ro_datatime(str_date):
    return datetime.strptime(str_date, '%Y/%m/%d')


print(x[1][0])
print(str_date_ro_datatime(x[1][0]))
print(type(str_date_ro_datatime(x[1][0]).year))


def timeToFloat(time):
    hr, min, sec = [float(x) for x in s.split(':')]
    return (hr * 3600 + min * 60 + sec) / 86400
