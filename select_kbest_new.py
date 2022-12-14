import csv
import numpy
import sklearn
from datetime import *

from sklearn.feature_selection import SelectKBest, f_regression


# Chargement des données
data = csv.reader(open("Occupancy_Estimation.csv", "r"),
                  delimiter=",")

data = list(data)
# delete la premiere ligne
data = numpy.delete(data, 0, 0)
#recupère les datas de la derniere colonne
target_data = data[:, -1]

# delete la derniere colonne : fait office de sortie
data = numpy.delete(data, -1, axis=1)


# On extraie la colonne des dates
dates = [i[0] for i in data]

# Permet de transformer un string en type date
def str_date_to_datatime(str_date):
    return datetime.strptime(str_date, '%Y/%m/%d')


# on transforme notre tableau de dates de type string en dates de type date
for i in range(0, int(len(dates))):
    dates[i] = str_date_to_datatime(dates[i])

# Permet de tranformer une date en int
def date_to_integer(date):
    return 10000 * date.year + 100 * date.month + date.day


# on transforme notre tableau de dates de type date en dates de type int
for i in range(0, int(len(dates))):
    dates[i] = date_to_integer(dates[i])

# Permet de transformer nos dates en nombre de jours effectifs sur lesquels les mesures sont effectuées
def date_to_days(tab_date):
    day = numpy.zeros(int(len(tab_date)))
    for i in range(1, int(len(tab_date))):
        if tab_date[i] > tab_date[i-1]:
            day[i] += day[i-1]+1
        else:
            day[i] = day[i-1]
    return day


# 6 jours de mesures : jours numérotés de 0 à 6
dates = date_to_days(dates)

# Donne l'heure de la journée normalisée de 0 à 1
def timeToFloat(time):
    hr, min, sec = [float(data) for data in time.split(':')]
    return (hr * 3600 + min * 60 + sec) / 86400


# On extraie la colonne des horaires
times = [i[1] for i in data]

# Rempli les heures normalisées de 0 à 1
for i in range(0, int(len(times))):
    times[i] = timeToFloat(times[i])
times = numpy.array(times)

# transforme notre tableau en numpy array
data = numpy.array(data)

# #delete les deux premieres colonnes car nous venons de transformer les types des dates et des horaires
data = numpy.delete(data, 0, 1)
data = numpy.delete(data, 0, 1)

# On crée un tableau de 0 avec 2 colones
dates_times = numpy.zeros((int(len(dates)), 2))

# On remplis le tableau avec les dates et les heures
for i in range(0, int(len(dates))):
    dates_times[i][0] = dates[i]
    dates_times[i][1] = times[i]

# on concatene les deudata tableaudata
data = numpy.concatenate((dates_times, data), axis=1)

# On transforme toutes les valeurs en float pour pouvoir les utiliser
data = data.astype(numpy.float)
# on normalise les données
data = sklearn.preprocessing.normalize(data)

#converti datas de sortie/cible en float
target_data = target_data.astype(numpy.float)
#selectionne les 3 colonnes les plus pertinentes
data = SelectKBest(f_regression, k=3).fit_transform(data, target_data)

