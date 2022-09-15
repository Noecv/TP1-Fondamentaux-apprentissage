import csv
import numpy
from datetime import *
reader = csv.reader(open("Occupancy_Estimation.csv", "r"),
                    delimiter=",")
x = list(reader)
dates=[i[0] for i in x]

def str_date_to_datatime(str_date):
    return datetime.strptime(str_date, '%Y/%m/%d')
for i in range(1, int(len(dates))):
    dates[i] = str_date_to_datatime(dates[i])

def date_to_integer(date):
    return 10000 * date.year + 100 * date.month + date.day
for i in range(1, int(len(dates))):
    dates[i] = date_to_integer(dates[i])

def date_to_days(tab_date):
    day=numpy.zeros(int(len(tab_date)))
    for i in range(2, int(len(tab_date))):
        if tab_date[i] > tab_date[i-1]: day[i] += day[i-1]+1
        else: day[i] = day[i-1]
    return day
dates = date_to_days(dates)

def timeToFloat(time):
    hr, min, sec = [float(x) for x in time.split(':')]
    return (hr * 3600 + min * 60 + sec) / 86400

times=[i[1] for i in x]
for i in range(1, int(len(times))):
    times[i] = timeToFloat(times[i])
times = numpy.array(times)

x = numpy.array(x)

x = numpy.delete(x, 0, 0)
x = numpy.delete(x, 0, 1)
x = numpy.delete(x, 0, 1)

dates_times = numpy.zeros((int(len(dates)), 2))
for i in range(1, int(len(dates))):
    dates_times[i][0] = dates[i]
    dates_times[i][1] = times[i]

dates_times = numpy.delete(dates_times, 0, 0)

x = numpy.concatenate((dates_times, x),axis=1)

(n, p) = numpy.shape(x)
Z = numpy.zeros((n, p))
print(type(Z))
for i in range(1, p):
    for j in range(1, n):
        #Z(j, i) = (X(j, i) - mu(:, i)) / (sigma(:, i));
        Z[j, i] = (x[j, i] - numpy.mean[:, i])/

