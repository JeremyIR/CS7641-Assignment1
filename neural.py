from sklearn.metrics import classification_report
from sklearn.model_selection import learning_curve, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier 
import numpy as np
import matplotlib.pyplot as plt
import time 
from processdata import XtrC, XteC, yTrC, yTeC, XtrG, XteG, yTrG, yTeG
plt.cla()

#reommended to scale the data
start = time.time()
scaler = StandardScaler()
scaler.fit(XtrC)
XtrC = scaler.transform(XtrC)
XteC = scaler.transform(XteC)

params = {'alpha': [0.001, 0.004, 0.006, 0.01, 0.04, 0.06] ,'hidden_layer_sizes': list(range(1,60,10))}
net = MLPClassifier(solver='lbfgs', max_iter=2500,early_stopping=True, random_state=42)
clf = GridSearchCV(net, param_grid=params, cv=3)
clf.fit(XtrC, yTrC)

train_sizes, train_scores, test_scores = learning_curve(clf.best_estimator_, XtrC, yTrC, scoring='accuracy', n_jobs=-1, train_sizes=np.linspace(0.01, 1.0, 100))
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
plt.xlabel("Training Set Size"), plt.ylabel("Accuracy"), plt.legend(loc="best")
plt.tight_layout()
plt.savefig('learning_curve_accuracy_neural_reduced.png')
plt.cla()
stop = time.time()
total_time = stop - start
print(total_time)
# print best parameter after tuning 
print(clf.best_params_) 
# print how our model looks after hyper-parameter tuning 
print(clf.best_estimator_) 
grid_predictions = clf.predict(XteC) 
# print classification report 
print(classification_report(yTeC, grid_predictions)) 

## GAMES 
startG = time.time()
plt.cla()
#reommended to scale the data
scalerG = StandardScaler()
scalerG.fit(XtrG)
XtrG = scalerG.transform(XtrG)
XteG = scalerG.transform(XteG)

paramsG = {'alpha': [0.001, 0.002, 0.004, 0.006, 0.01, 0.02, 0.04, 0.06] ,'hidden_layer_sizes': list(range(1,60,5))}
netG = MLPClassifier(solver='lbfgs', max_iter=10000,early_stopping=True, random_state=42)
clfG = GridSearchCV(netG, param_grid=paramsG, cv=3)
clfG.fit(XtrG, yTrG)

train_sizesG, train_scoresG, test_scoresG = learning_curve(clfG.best_estimator_, XtrG, yTrG, scoring='accuracy', n_jobs=-1, train_sizes=np.linspace(0.01, 1.0, 100))
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
plt.xlabel("Training Set Size"), plt.ylabel("Accuracy"), plt.legend(loc="best")
plt.tight_layout()
plt.savefig('learning_curve_accuracy_neural_game.png')
plt.cla()
stopG = time.time()
total_timeG = stopG - startG
print(total_timeG)
# print best parameter after tuning 
print(clfG.best_params_) 

# print how our model looks after hyper-parameter tuning 
print(clfG.best_estimator_) 

grid_predictionsG = clfG.predict(XteG) 
  
# print classification report 
print(classification_report(yTeG, grid_predictionsG)) 