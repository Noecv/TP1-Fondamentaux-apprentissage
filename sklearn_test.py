import csv
import numpy
import sklearn
from datetime import *
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
reader = csv.reader(open("Occupancy_Estimation.csv", "r"),
                    delimiter=",")
x = list(reader)
# delete la premiere ligne
# delete la derniere colonne : fait office de sortie
x = numpy.delete(x, -1, axis=1)
x = numpy.delete(x, 0, 0)

# On extraie la colonne des dates
dates = [i[0] for i in x]

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
    hr, min, sec = [float(x) for x in time.split(':')]
    return (hr * 3600 + min * 60 + sec) / 86400


# On extraie la colonne des horaires
times = [i[1] for i in x]

# Rempli les heures normalisées de 0 à 1
for i in range(0, int(len(times))):
    times[i] = timeToFloat(times[i])
times = numpy.array(times)

# transforme notre tableau en numpy array
x = numpy.array(x)

# #delete les deux premieres colonnes car nous venons de transformer les types des dates et des horaires
x = numpy.delete(x, 0, 1)
x = numpy.delete(x, 0, 1)

# On créer un tableau de 0 avec 2 colones
dates_times = numpy.zeros((int(len(dates)), 2))

# On remplis le tableau avec les dates et les heures
for i in range(0, int(len(dates))):
    dates_times[i][0] = dates[i]
    dates_times[i][1] = times[i]

# on concatene les deux tableaux
x = numpy.concatenate((dates_times, x), axis=1)

# Désormais nous pouvons passer au PCA

#################################### PCA avec sklearn #################################################################
sc = StandardScaler()
skx1 = sc.fit_transform(x)
pca = PCA(n_components=3, svd_solver='full')
skx2 = pca.fit_transform(skx1)


#################################### PCA Fait à la main #################################################################

# On transforme notre tableau numpy en tableau de float
x = x.astype(float)
# on normalise les données
x = (x - x.mean(axis=0)) / x.std(axis=0)

# On vérifie que les données sont bien normalisées de la même manière
print("fit identiques : ", numpy.all(numpy.equal(skx1, x)))

# On créer un tableau de 0 avec 3 colones
(n, p) = numpy.shape(x)
Z = numpy.zeros((n, p))

# On calcule la covariance de x et on obtient une matrice symétrique
covx = numpy.cov(x, rowvar=False)


# On récupère les valeurs propres et les vecteurs propres de la matrice de covariance necessaire pour le PCA
valeurs_propres, vecteurs_propres = numpy.linalg.eig(covx)

# On créér un tableau de zero avant de le remplire avec les données dont les valeurs propres respectent le critère de kaiser
# C'est une matrice des vecteurs propres dont les valeurs propres sont supérieures à 1
FeatureVector = numpy.zeros((0, p))
for i in range(0, len(valeurs_propres)):
    if valeurs_propres[i] > 1:
        FeatureVector = numpy.concatenate(
            (FeatureVector, [vecteurs_propres[:, i]]), axis=0)

# On fait la multiplcation matricielle entre x et la matrice transposée des vecteurs propres afin d'avoir notre nouveau set de données
Xnew = numpy.dot(x, numpy.transpose(FeatureVector))
#Xnew = numpy.transpose(FeatureVector)*x

print("Xnew : ", Xnew)
print("Xnew shape : ", numpy.shape(Xnew))

#################################### Comparaison entre sklearn et notre algo #################################################################

print("xnew sklearn :", skx2)
print("Feature vector :", FeatureVector)
print("Feature vector sklearn :", pca.components_)
print("Feature vector identiques : ",
      numpy.equal(pca.components_, FeatureVector))
print("matrices finales identiques : ", numpy.equal(skx2, Xnew))

# Les prints renvoient false à cause des arrondis de float
