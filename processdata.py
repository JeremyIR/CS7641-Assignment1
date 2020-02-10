import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

le=LabelEncoder()

## car dataset
car = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data', sep=',', header = None)
car.columns = ['buying','maint','doors','persons','lug_boot','safety','class']

plt.figure(figsize=(12,6))
sns.countplot(car['class'])
#plt.savefig('carClass.png')

for i in car.columns:
    car[i]=le.fit_transform(car[i])

XC=car[car.columns[:-1]]
yC=car['class']

XtrC, XteC, yTrC, yTeC = train_test_split(XC, yC, test_size=0.30, random_state=42)

#tic-tac-toe endgame dataset 
game = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/tic-tac-toe/tic-tac-toe.data', sep=',', header = None)
game.columns = ['topL','topM','topR','midL','midM','midR','botL','botM','botR','class']
'''for i in game.columns:
    print(game[i].value_counts())'''

plt.figure(figsize=(12,6))
sns.countplot(game[i],hue=game['class'])
#plt.savefig('gameClass.png')

sns.countplot(game['class'])
for i in game.columns[:-1]:
    plt.figure(figsize=(12,6))
    plt.title("For feature '%s'"%i)
    sns.countplot(game[i],hue=game['class'])
    #plt.savefig('feature' + i + '.png')

for i in game.columns:
    game[i]=le.fit_transform(game[i])

XG=game[game.columns[:-1]]
yG=game['class']

XtrG, XteG, yTrG, yTeG = train_test_split(XG, yG, test_size=0.30, random_state=42)

#validation curve for svm car
plt.cla()
data = ['C=1,gamma=0.1,kernel=linear,score=0.740','C=1,gamma=0.1,kernel=linear,score=0.731','C=1,gamma=0.1,kernel=linear,score=0.715','C=1,gamma=0.1,kernel=linear,score=0.723','C=1,gamma=0.1,kernel=linear,score=0.722','C=1,gamma=0.1,kernel=poly,score=0.793','C=1,gamma=0.1,kernel=poly,score=0.748','C=1,gamma=0.1,kernel=poly,score=0.777','C=1,gamma=0.1,kernel=poly,score=0.781','C=1,gamma=0.1,kernel=poly,score=0.793','C=1,gamma=0,kernel=linear,score=0.740','C=1,gamma=0,kernel=linear,score=0.731','C=1,gamma=0,kernel=linear,score=0.715','C=1,gamma=0,kernel=linear,score=0.723','C=1,gamma=0,kernel=linear,score=0.722','C=1,gamma=0,kernel=poly,score=0.702','C=1,gamma=0,kernel=poly,score=0.707','C=1,gamma=0,kernel=poly,score=0.707','C=1,gamma=0,kernel=poly,score=0.702','C=1,gamma=0,kernel=poly,score=0.705','C=1,gamma=3,kernel=linear,score=0.740','C=1,gamma=3,kernel=linear,score=0.731','C=1,gamma=3,kernel=linear,score=0.715','C=1,gamma=3,kernel=linear,score=0.723','C=1,gamma=3,kernel=linear,score=0.722','C=1,gamma=3,kernel=poly,score=0.851','C=1,gamma=3,kernel=poly,score=0.847','C=1,gamma=3,kernel=poly,score=0.855','C=1,gamma=3,kernel=poly,score=0.860','C=1,gamma=3,kernel=poly,score=0.871','C=1,gamma=0.5,kernel=linear,score=0.740','C=1,gamma=0.5,kernel=linear,score=0.731','C=1,gamma=0.5,kernel=linear,score=0.715','C=1,gamma=0.5,kernel=linear,score=0.723','C=1,gamma=0.5,kernel=linear,score=0.722','C=1,gamma=0.5,kernel=poly,score=0.872','C=1,gamma=0.5,kernel=poly,score=0.860','C=1,gamma=0.5,kernel=poly,score=0.897','C=1,gamma=0.5,kernel=poly,score=0.860','C=1,gamma=0.5,kernel=poly,score=0.884,','C=1,gamma=0.6,kernel=linear,score=0.740','C=1,gamma=0.6,kernel=linear,score=0.731','C=1,gamma=0.6,kernel=linear,score=0.715','C=1,gamma=0.6,kernel=linear,score=0.723','C=1,gamma=0.6,kernel=linear,score=0.722','C=1,gamma=0.6,kernel=poly,score=0.847','C=1,gamma=0.6,kernel=poly,score=0.855','C=1,gamma=0.6,kernel=poly,score=0.893','C=1,gamma=0.6,kernel=poly,score=0.851','C=1,gamma=0.6,kernel=poly,score=0.855','C=3,gamma=0.1,kernel=linear,score=0.744','C=3,gamma=0.1,kernel=linear,score=0.736','C=3,gamma=0.1,kernel=linear,score=0.715','C=3,gamma=0.1,kernel=linear,score=0.727','C=3,gamma=0.1,kernel=linear,score=0.726','C=3,gamma=0.1,kernel=poly,score=0.843','C=3,gamma=0.1,kernel=poly,score=0.810','C=3,gamma=0.1,kernel=poly,score=0.839','C=3,gamma=0.1,kernel=poly,score=0.806','C=3,gamma=0.1,kernel=poly,score=0.830','C=3,gamma=0,kernel=linear,score=0.744','C=3,gamma=0,kernel=linear,score=0.736','C=3,gamma=0,kernel=linear,score=0.715','C=3,gamma=0,kernel=linear,score=0.727','C=3,gamma=0,kernel=linear,score=0.726','C=3,gamma=0,kernel=poly,score=0.702','C=3,gamma=0,kernel=poly,score=0.707','C=3,gamma=0,kernel=poly,score=0.707','C=3,gamma=0,kernel=poly,score=0.702','C=3,gamma=0,kernel=poly,score=0.705','C=3,gamma=3,kernel=linear,score=0.744','C=3,gamma=3,kernel=linear,score=0.736','C=3,gamma=3,kernel=linear,score=0.715','C=3,gamma=3,kernel=linear,score=0.727','C=3,gamma=3,kernel=linear,score=0.726','C=3,gamma=3,kernel=poly,score=0.855','C=3,gamma=3,kernel=poly,score=0.847','C=3,gamma=3,kernel=poly,score=0.855','C=3,gamma=3,kernel=poly,score=0.851','C=3,gamma=3,kernel=poly,score=0.871','C=3,gamma=0.5,kernel=linear,score=0.744','C=3,gamma=0.5,kernel=linear,score=0.736','C=3,gamma=0.5,kernel=linear,score=0.715','C=3,gamma=0.5,kernel=linear,score=0.727','C=3,gamma=0.5,kernel=linear,score=0.726','C=3,gamma=0.5,kernel=poly,score=0.839','C=3,gamma=0.5,kernel=poly,score=0.839','C=3,gamma=0.5,kernel=poly,score=0.876','C=3,gamma=0.5,kernel=poly,score=0.847','C=3,gamma=0.5,kernel=poly,score=0.859','C=3,gamma=0.6,kernel=linear,score=0.744','C=3,gamma=0.6,kernel=linear,score=0.736','C=3,gamma=0.6,kernel=linear,score=0.715','C=3,gamma=0.6,kernel=linear,score=0.727','C=3,gamma=0.6,kernel=linear,score=0.726','C=3,gamma=0.6,kernel=poly,score=0.843','C=3,gamma=0.6,kernel=poly,score=0.855','C=3,gamma=0.6,kernel=poly,score=0.868','C=3,gamma=0.6,kernel=poly,score=0.851','C=3,gamma=0.6,kernel=poly,score=0.863','C=5,gamma=0.1,kernel=linear,score=0.740','C=5,gamma=0.1,kernel=linear,score=0.736','C=5,gamma=0.1,kernel=linear,score=0.715','C=5,gamma=0.1,kernel=linear,score=0.727','C=5,gamma=0.1,kernel=linear,score=0.726','C=5,gamma=0.1,kernel=poly,score=0.843','C=5,gamma=0.1,kernel=poly,score=0.810','C=5,gamma=0.1,kernel=poly,score=0.847','C=5,gamma=0.1,kernel=poly,score=0.822','C=5,gamma=0.1,kernel=poly,score=0.851','C=5,gamma=0,kernel=linear,score=0.740','C=5,gamma=0,kernel=linear,score=0.736','C=5,gamma=0,kernel=linear,score=0.715','C=5,gamma=0,kernel=linear,score=0.727','C=5,gamma=0,kernel=linear,score=0.726','C=5,gamma=0,kernel=poly,score=0.702','C=5,gamma=0,kernel=poly,score=0.707','C=5,gamma=0,kernel=poly,score=0.707','C=5,gamma=0,kernel=poly,score=0.702','C=5,gamma=0,kernel=poly,score=0.705','C=5,gamma=3,kernel=linear,score=0.740','C=5,gamma=3,kernel=linear,score=0.736','C=5,gamma=3,kernel=linear,score=0.715','C=5,gamma=3,kernel=linear,score=0.727','C=5,gamma=3,kernel=linear,score=0.726','C=5,gamma=3,kernel=poly,score=0.855','C=5,gamma=3,kernel=poly,score=0.843','C=5,gamma=3,kernel=poly,score=0.855','C=5,gamma=3,kernel=poly,score=0.851','C=5,gamma=3,kernel=poly,score=0.876','C=5,gamma=0.5,kernel=linear,score=0.740','C=5,gamma=0.5,kernel=linear,score=0.736','C=5,gamma=0.5,kernel=linear,score=0.715','C=5,gamma=0.5,kernel=linear,score=0.727','C=5,gamma=0.5,kernel=linear,score=0.726','C=5,gamma=0.5,kernel=poly,score=0.843','C=5,gamma=0.5,kernel=poly,score=0.860','C=5,gamma=0.5,kernel=poly,score=0.872','C=5,gamma=0.5,kernel=poly,score=0.855','C=5,gamma=0.5,kernel=poly,score=0.867','C=5,gamma=0.6,kernel=linear,score=0.740','C=5,gamma=0.6,kernel=linear,score=0.736','C=5,gamma=0.6,kernel=linear,score=0.715','C=5,gamma=0.6,kernel=linear,score=0.727','C=5,gamma=0.6,kernel=linear,score=0.726','C=5,gamma=0.6,kernel=poly,score=0.851','C=5,gamma=0.6,kernel=poly,score=0.855','C=5,gamma=0.6,kernel=poly,score=0.855','C=5,gamma=0.6,kernel=poly,score=0.835','C=5,gamma=0.6,kernel=poly,score=0.871','C=10,gamma=0.1,kernel=linear,score=0.744','C=10,gamma=0.1,kernel=linear,score=0.736','C=10,gamma=0.1,kernel=linear,score=0.715','C=10,gamma=0.1,kernel=linear,score=0.727','C=10,gamma=0.1,kernel=linear,score=0.730','C=10,gamma=0.1,kernel=poly,score=0.876','C=10,gamma=0.1,kernel=poly,score=0.810','C=10,gamma=0.1,kernel=poly,score=0.855','C=10,gamma=0.1,kernel=poly,score=0.826','C=10,gamma=0.1,kernel=poly,score=0.859','C=10,gamma=0,kernel=linear,score=0.744','C=10,gamma=0,kernel=linear,score=0.736','C=10,gamma=0,kernel=linear,score=0.715','C=10,gamma=0,kernel=linear,score=0.727','C=10,gamma=0,kernel=linear,score=0.730','C=10,gamma=0,kernel=poly,score=0.702','C=10,gamma=0,kernel=poly,score=0.707','C=10,gamma=0,kernel=poly,score=0.707','C=10,gamma=0,kernel=poly,score=0.702','C=10,gamma=0,kernel=poly,score=0.705','C=10,gamma=3,kernel=linear,score=0.744','C=10,gamma=3,kernel=linear,score=0.736','C=10,gamma=3,kernel=linear,score=0.715','C=10,gamma=3,kernel=linear,score=0.727','C=10,gamma=3,kernel=linear,score=0.730','C=10,gamma=3,kernel=poly,score=0.855','C=10,gamma=3,kernel=poly,score=0.843','C=10,gamma=3,kernel=poly,score=0.855','C=10,gamma=3,kernel=poly,score=0.851','C=10,gamma=3,kernel=poly,score=0.876','C=10,gamma=0.5,kernel=linear,score=0.744','C=10,gamma=0.5,kernel=linear,score=0.736','C=10,gamma=0.5,kernel=linear,score=0.715','C=10,gamma=0.5,kernel=linear,score=0.727','C=10,gamma=0.5,kernel=linear,score=0.730','C=10,gamma=0.5,kernel=poly,score=0.855','C=10,gamma=0.5,kernel=poly,score=0.851','C=10,gamma=0.5,kernel=poly,score=0.851','C=10,gamma=0.5,kernel=poly,score=0.835','C=10,gamma=0.5,kernel=poly,score=0.876','C=10,gamma=0.6,kernel=linear,score=0.744','C=10,gamma=0.6,kernel=linear,score=0.736','C=10,gamma=0.6,kernel=linear,score=0.715','C=10,gamma=0.6,kernel=linear,score=0.727','C=10,gamma=0.6,kernel=linear,score=0.730','C=10,gamma=0.6,kernel=poly,score=0.860','C=10,gamma=0.6,kernel=poly,score=0.843','C=10,gamma=0.6,kernel=poly,score=0.855','C=10,gamma=0.6,kernel=poly,score=0.839','C=10,gamma=0.6,kernel=poly,score=0.896']
C = ([int(x.split(',')[0][2:]) for x in data])
joined = ([[float(x.split(',')[1][6:]),[x.split(',')[2][7:]][0]] for x in data])
scores = ([float(x.split(',')[3][6:]) for x in data])
for ind, i in enumerate(joined):
    plt.plot(C, scores, label='Kernel: ' + str(i[1]) + ', Gamma: ' + str(i[0]))
plt.legend(fontsize='xx-small', loc='right', bbox_to_anchor=(0.5, 0., 0.5, 0.5))
plt.xlabel('C')
plt.ylabel('Accuracy')
plt.title("Validation Curve")
plt.savefig('validation_svm_car.png')
plt.cla()

