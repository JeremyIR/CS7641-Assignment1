from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import learning_curve, validation_curve, cross_val_score, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt
import time 
from processdata import XtrC, XteC, yTrC, yTeC, XtrG, XteG, yTrG, yTeG
plt.cla()

start = time.time()
knn=KNeighborsClassifier(n_jobs=-1)
neighbors = list(range(2,40))
params = {'n_neighbors': neighbors}
clf = GridSearchCV(knn, params)

clf.fit(XtrC,yTrC)
train_prediction = clf.predict(XtrC)
test_prediction = clf.predict(XteC)

#accuracy
#train_sizes, train_scores, test_scores = learning_curve(clf.best_estimator_, XtrC, yTrC, scoring='accuracy', n_jobs=-1, train_sizes=np.linspace(0.01, 1.0, 100))
# f1
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
#plt.savefig('learning_curve_accuracy_knn.png')
# f1 
plt.savefig('learning_curve_accuracy_knn_f1.png')

plt.cla()
stop = time.time()
total_time = stop - start
print(total_time)

train_classification_report = classification_report(yTrC, train_prediction)
test_classification_report = classification_report(yTeC, test_prediction)
print(train_classification_report)
print(test_classification_report)
print(clf.best_score_)
print(clf.best_params_)

scores = []
for k in range(2,40):
    clf=KNeighborsClassifier(n_jobs=-1,n_neighbors=k)
    #score=cross_val_score(clf,XtrC,yTrC,cv=5,n_jobs=-1,scoring='accuracy')
    # f1
    score=cross_val_score(clf,XtrC,yTrC,cv=5,n_jobs=-1,scoring='f1_weighted')
    scores.append(score.mean())

plt.figure(figsize=(12,8))
plt.plot(range(2,40),scores)
plt.title("Validation Curve")
plt.xlabel("n_neighbours")
# plt.ylabel("accuracy")
# plt.savefig('acc_vs_neighbors.png')
plt.ylabel("F1 Score")
plt.savefig('acc_vs_neighbors_f1.png')
plt.cla()

######## game
start_game = time.time()

knn_game=KNeighborsClassifier(n_jobs=-1)
neighbors_game = list(range(2,40))
params_game = {'n_neighbors': neighbors_game}
clf_game = GridSearchCV(knn_game, params_game)

clf_game.fit(XtrG,yTrG)
train_prediction_game = clf_game.predict(XtrG)
test_prediction_game = clf_game.predict(XteG)

#accuracy
#train_sizes_game, train_scores_game, test_scores_game = learning_curve(clf_game.best_estimator_, XtrG, yTrG, scoring='accuracy', n_jobs=-1, train_sizes=np.linspace(0.01, 1.0, 100))
# f1
train_sizes_game, train_scores_game, test_scores_game = learning_curve(clf_game.best_estimator_, XtrG, yTrG, scoring='f1_weighted', n_jobs=-1, train_sizes=np.linspace(0.01, 1.0, 100))

# Create means and standard deviations of training set scores
train_mean_game = np.mean(train_scores_game, axis=1)
train_std_game = np.std(train_scores_game, axis=1)
# Create means and standard deviations of test set scores
test_mean_game = np.mean(test_scores_game, axis=1)
test_std_game = np.std(test_scores_game, axis=1)
# Draw lines
plt.plot(train_sizes_game, train_mean_game, '--', color="#111111",  label="Training score")
plt.plot(train_sizes_game, test_mean_game, color="#111111", label="Testing score")
# Draw bands
plt.fill_between(train_sizes_game, train_mean_game - train_std_game, train_mean_game + train_std_game, color="#DDDDDD")
plt.fill_between(train_sizes_game, test_mean_game - test_std_game, test_mean_game + test_std_game, color="#DDDDDD")
# Create plot
plt.title("Learning Curve")
#plt.xlabel("Training Set Size"), plt.ylabel("Accuracy"), plt.legend(loc="best")
# f1
plt.xlabel("Training Set Size"), plt.ylabel("F1 Score"), plt.legend(loc="best")
plt.tight_layout()
#plt.savefig('learning_curve_accuracy_knn_game.png')
# f1 
plt.savefig('learning_curve_accuracy_knn_game_f1.png')

plt.cla()
stop_game = time.time()
total_time_game = stop_game - start_game
print(total_time_game)

train_classification_report_game = classification_report(yTrG, train_prediction_game)
test_classification_report_game = classification_report(yTeG, test_prediction_game)
print(train_classification_report_game)
print(test_classification_report_game)
print(clf_game.best_score_)
print(clf_game.best_params_)

avg_game = []
for k in range(2,40):
    clf_game=KNeighborsClassifier(n_jobs=-1,n_neighbors=k)
    #score=cross_val_score(clf_game,XtrG,yTrG,cv=5,n_jobs=-1,scoring='accuracy')
    score=cross_val_score(clf_game,XtrG,yTrG,cv=5,n_jobs=-1,scoring='f1_weighted')
    avg_game.append(score.mean())

plt.figure(figsize=(12,8))
plt.plot(range(2,40),avg_game)
plt.title("Validation Curve")
plt.xlabel("n_neighbours")
#plt.ylabel("accuracy")
#plt.savefig('acc_vs_neighbors_game.png')
plt.ylabel("F1 Score")
plt.savefig('acc_vs_neighbors_game_f1.png')
plt.cla()