import csv
import numpy
from datetime import *
reader = csv.reader(open("Occupancy_Estimation.csv", "r"),
                    delimiter=",")
x = list(reader)
dates = [i[0] for i in x]


def str_date_to_datatime(str_date):
    return datetime.strptime(str_date, '%Y/%m/%d')


for i in range(1, int(len(dates))):
    dates[i] = str_date_to_datatime(dates[i])


def date_to_integer(date):
    return 10000 * date.year + 100 * date.month + date.day


for i in range(1, int(len(dates))):
    dates[i] = date_to_integer(dates[i])


def date_to_days(tab_date):
    day = numpy.zeros(int(len(tab_date)))
    for i in range(2, int(len(tab_date))):
        if tab_date[i] > tab_date[i-1]:
            day[i] += day[i-1]+1
        else:
            day[i] = day[i-1]
    return day


dates = date_to_days(dates)


def timeToFloat(time):
    hr, min, sec = [float(x) for x in time.split(':')]
    return (hr * 3600 + min * 60 + sec) / 86400


times = [i[1] for i in x]
for i in range(1, int(len(times))):
    times[i] = timeToFloat(times[i])

times = numpy.array(times, ndim=2)
print(numpy.concatenate((times.T, dates.T), axis=0))

# concatener puis ajouter au gros tableau avec le reste des données
