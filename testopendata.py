import csv
import numpy
from datetime import *
reader = csv.reader(open("Occupancy_Estimation.csv", "r"),
                    delimiter=",")
x = list(reader)

x = numpy.delete(x, -1, axis=1)
print(numpy.shape(x))
# ligne servant à limiter la taille des données (donc des calculs) lors des tests
#x = x[0:2000]
print(numpy.shape(x))
# On extraie la colonne des dates
dates = [i[0] for i in x]

# Permet de transformer un string en type date


def str_date_to_datatime(str_date):
    return datetime.strptime(str_date, '%Y/%m/%d')


# on transforme notre tableau de dates de type string en dates de type date
for i in range(1, int(len(dates))):
    dates[i] = str_date_to_datatime(dates[i])

# Permet de tranformer une date en int


def date_to_integer(date):
    return 10000 * date.year + 100 * date.month + date.day


# on transforme notre tableau de dates de type date en dates de type int
for i in range(1, int(len(dates))):
    dates[i] = date_to_integer(dates[i])

# Permet de transformer nos dates en nombre de jours effectifs sur lesquels les mesures sont effectuées


def date_to_days(tab_date):
    day = numpy.zeros(int(len(tab_date)))
    for i in range(2, int(len(tab_date))):
        if tab_date[i] > tab_date[i-1]:
            day[i] += day[i-1]+1
        else:
            day[i] = day[i-1]
    return day


# 6 jours de mesures : jours numérotés de 0 à 6
dates = date_to_days(dates)

# Donne l'heure de la journée normalisée de 0 à 1


def timeToFloat(time):
    hr, min, sec = [float(x) for x in time.split(':')]
    return (hr * 3600 + min * 60 + sec) / 86400


# On extraie la colonne des horaires
times = [i[1] for i in x]

# Rempli les heures normalisées de 0 à 1
for i in range(1, int(len(times))):
    times[i] = timeToFloat(times[i])
times = numpy.array(times)

# transforme notre tableau en numpy array
x = numpy.array(x)
# delete la premiere ligne, et les deux premieres colonnes car nous venons de transformer les types des dates et des horaires
x = numpy.delete(x, 0, 0)
x = numpy.delete(x, 0, 1)
x = numpy.delete(x, 0, 1)

# On créer un tableau de 0 avec 2 colones
dates_times = numpy.zeros((int(len(dates)), 2))

# On remplis le tableau avec les dates et les heures
for i in range(1, int(len(dates))):
    dates_times[i][0] = dates[i]
    dates_times[i][1] = times[i]

# on supprime la première ligne qui est vide
dates_times = numpy.delete(dates_times, 0, 0)

# on concatene les deux tableaux
x = numpy.concatenate((dates_times, x), axis=1)

# On transforme toutes les valeurs en float pour pouvoir les utiliser
x = x.astype(numpy.float)
# on normalise les données
x = (x - x.mean(axis=0)) / x.std(axis=0)

(n, p) = numpy.shape(x)
Z = numpy.zeros((n, p))


# On calcule la covariance de x et on obtient une matrice symétrique
covx = numpy.cov(x, rowvar=False)


# eigen value of covx
valeurs_propres, vecteurs_propres = numpy.linalg.eig(covx)

#numpy.extract(vecteurs_propres < 1, vecteurs_propres)

# calculate featureVector of eigenvalue


FeatureVector = numpy.zeros((0, p))

for i in range(0, len(valeurs_propres)):
    if valeurs_propres[i] > 1:
        print(i)
        FeatureVector = numpy.concatenate(
            (FeatureVector, [vecteurs_propres[i]]), axis=0)


print(numpy.shape(FeatureVector))
print(FeatureVector)
Xnew = numpy.dot(x, numpy.transpose(FeatureVector))
#Xnew = numpy.transpose(FeatureVector)*x

print("Xnew : ", Xnew)
print("Xnew shape : ", numpy.shape(Xnew))

# a comparer avec slecetkbest de sklearn
