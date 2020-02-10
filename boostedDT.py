from processdata import XtrC, XteC, yTrC, yTeC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import learning_curve, cross_val_score, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
import numpy as np
import matplotlib.pyplot as plt
import time
from processdata import XtrC, XteC, yTrC, yTeC, XtrG, XteG, yTrG, yTeG
plt.cla()

#### CARS
start = time.time()
dT = DecisionTreeClassifier(random_state=42, ccp_alpha=0.04)
estimators = list(range(1,400,10))
params = {'n_estimators': estimators}
boost = AdaBoostClassifier(base_estimator=dT, random_state=42)
clf = GridSearchCV(boost, params)
clf.fit(XtrC, yTrC)

train_prediction = clf.predict(XtrC)
test_prediction = clf.predict(XteC)

#train_sizes, train_scores, test_scores = learning_curve(clf.best_estimator_, XtrC, yTrC, scoring='accuracy', n_jobs=-1, train_sizes=np.linspace(0.01, 1.0, 100))
#F1
train_sizes, train_scores, test_scores = learning_curve(clf.best_estimator_, XtrC, yTrC, scoring='f1_weighted', n_jobs=-1, train_sizes=np.linspace(0.01, 1.0, 100))
# Create means and standard deviations of training set scores
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
# Create means and standard deviations of test set scores
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)
# Draw lines
plt.plot(train_sizes, train_mean, '--', color="#111111",  label="Training score")
plt.plot(train_sizes, test_mean, color="#111111", label="Testing score")
# Draw bands
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color="#DDDDDD")
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color="#DDDDDD")
# Create plot
plt.title("Learning Curve")
#plt.xlabel("Training Set Size"), plt.ylabel("Accuracy"), plt.legend(loc="best")
# f1
plt.xlabel("Training Set Size"), plt.ylabel("F1 Score"), plt.legend(loc="best")
plt.tight_layout()
#plt.savefig('learning_curve_accuracy_boost.png')
# F1 
plt.savefig('learning_curve_accuracy_boost_f1.png')
plt.cla()
stop = time.time()
total_time = stop - start
print(total_time)

train_classification_report = classification_report(yTrC, train_prediction)
test_classification_report = classification_report(yTeC, test_prediction)
#print(train_classification_report)
print(test_classification_report)
print(clf.best_score_)
print(clf.best_params_)

avg = []
for k in range(1,400,10):
    clf=AdaBoostClassifier(base_estimator=dT, random_state=42, n_estimators=k)
    #score=cross_val_score(clf,XtrC,yTrC,cv=5,n_jobs=-1,scoring='accuracy')
    #F1
    score=cross_val_score(clf,XtrC,yTrC,cv=5,n_jobs=-1,scoring='f1_weighted')
    avg.append(score.mean())

plt.figure(figsize=(12,8))
plt.plot(range(1,400,10),avg)
plt.title("Validation Curve")
plt.xlabel("n_estimators")
# plt.ylabel("accuracy")
# plt.savefig('acc_vs_estimators.png')
# f1
plt.ylabel("F1 Score")
plt.savefig('acc_vs_estimators_f1.png')

plt.cla()

### GAMES
startG = time.time()
plt.cla()

dTG = DecisionTreeClassifier(random_state=42, ccp_alpha=0.016)
estimatorsG = list(range(1,400,10))
paramsG = {'n_estimators': estimatorsG}
boostG = AdaBoostClassifier(base_estimator=dTG, random_state=42)
clfG = GridSearchCV(boostG, paramsG)
clfG.fit(XtrG, yTrG)

train_predictionG = clfG.predict(XtrG)
test_predictionG = clfG.predict(XteG)

#train_sizesG, train_scoresG, test_scoresG = learning_curve(clfG.best_estimator_, XtrG, yTrG, scoring='accuracy', n_jobs=-1, train_sizes=np.linspace(0.01, 1.0, 100))
# f1
train_sizesG, train_scoresG, test_scoresG = learning_curve(clfG.best_estimator_, XtrG, yTrG, scoring='f1_weighted', n_jobs=-1, train_sizes=np.linspace(0.01, 1.0, 100))

# Create means and standard deviations of training set scores
train_meanG = np.mean(train_scoresG, axis=1)
train_stdG = np.std(train_scoresG, axis=1)
# Create means and standard deviations of test set scores
test_meanG = np.mean(test_scoresG, axis=1)
test_stdG = np.std(test_scoresG, axis=1)
# Draw lines
plt.plot(train_sizesG, train_meanG, '--', color="#111111",  label="Training score")
plt.plot(train_sizesG, test_meanG, color="#111111", label="Testing score")
# Draw bands
plt.fill_between(train_sizesG, train_meanG - train_stdG, train_meanG + train_stdG, color="#DDDDDD")
plt.fill_between(train_sizesG, test_meanG - test_stdG, test_meanG + test_stdG, color="#DDDDDD")
# Create plot
plt.title("Learning Curve")
# plt.xlabel("Training Set Size"), plt.ylabel("Accuracy"), plt.legend(loc="best")
# f1 
plt.xlabel("Training Set Size"), plt.ylabel("F1 Score"), plt.legend(loc="best")
plt.tight_layout()
# plt.savefig('learning_curve_accuracy_boost_game.png')
# f1
plt.savefig('learning_curve_accuracy_boost_game_f1.png')

plt.cla()
stopG = time.time()
total_timeG = stopG - startG
print(total_timeG)

train_classification_reportG = classification_report(yTrG, train_predictionG)
test_classification_reportG = classification_report(yTeG, test_predictionG)
#print(train_classification_reportG)
print(test_classification_reportG)
print(clfG.best_score_)
print(clfG.best_params_)

avgG = []
for k in range(1,400,10):
    clf=AdaBoostClassifier(base_estimator=dTG, random_state=42, n_estimators=k)
    #score=cross_val_score(clf,XtrG,yTrG,cv=5,n_jobs=-1,scoring='accuracy')
    # F1
    score=cross_val_score(clf,XtrG,yTrG,cv=5,n_jobs=-1,scoring='f1_weighted')
    avgG.append(score.mean())

plt.figure(figsize=(12,8))
plt.plot(range(1,400,10),avgG)
plt.title("Validation Curve")
plt.xlabel("n_estimators")
plt.ylabel("F1 Score")
plt.savefig('acc_vs_estimators_game_F1.png')
# plt.ylabel("accuracy")
# plt.savefig('acc_vs_estimators_game.png')
# F1

plt.cla()