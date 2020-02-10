from processdata import XtrC, XteC, yTrC, yTeC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import learning_curve, cross_val_score, GridSearchCV, validation_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC 
import numpy as np
import matplotlib.pyplot as plt
import time
from processdata import XtrC, XteC, yTrC, yTeC, XtrG, XteG, yTrG, yTeG
plt.cla()


## Games
startG = time.time()
# defining parameter range 
param_gridG = {'C': [1,3,5,10],  
              'gamma': [0.1, 0,3, 0.5, 0.6] ,
              'kernel': ['linear','poly']}  
              #'kernel': ['rbf', 'linear']}  
  
gridG = GridSearchCV(SVC(random_state=42), param_gridG, refit = True, verbose = 3) 
  
# fitting the model for grid search 
gridG.fit(XtrG, yTrG) 

train_sizesG, train_scoresG, test_scoresG = learning_curve(gridG.best_estimator_, XtrG, yTrG, scoring='accuracy', n_jobs=-1, train_sizes=np.linspace(0.01, 1.0, 100))
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
plt.savefig('learning_curve_accuracy_svm_Game.png')
plt.cla()
stopG = time.time()
total_timeG = stopG - startG
# print best parameter after tuning 
print(gridG.best_params_) 
# print how our model looks after hyper-parameter tuning 
print(gridG.best_estimator_) 
gridG_predictions = gridG.predict(XteG) 
# print classification report 
print(classification_report(yTeG, gridG_predictions)) 

print(total_timeG)

train_sizesG_F1, train_scoresG_F1, test_scoresG_f1 = learning_curve(gridG.best_estimator_, XtrG, yTrG, scoring='f1_weighted', n_jobs=-1, train_sizes=np.linspace(0.01, 1.0, 100))
# Create means and standard deviations of training set scores
train_meanG_F1 = np.mean(train_scoresG_F1, axis=1)
train_stdG_F1 = np.std(train_scoresG_F1, axis=1)
# Create means and standard deviations of test set scores
test_meanG_F1 = np.mean(test_scoresG_f1, axis=1)
test_stdG_F1 = np.std(test_scoresG_f1, axis=1)
# Draw lines
plt.plot(train_sizesG_F1, train_meanG_F1, '--', color="#111111",  label="Training score")
plt.plot(train_sizesG_F1, test_meanG_F1, color="#111111", label="Testing score")
# Draw bands
plt.fill_between(train_sizesG_F1, train_meanG_F1 - train_stdG_F1, train_meanG_F1 + train_stdG_F1, color="#DDDDDD")
plt.fill_between(train_sizesG_F1, test_meanG_F1 - test_stdG_F1, test_meanG_F1 + test_stdG_F1, color="#DDDDDD")
# Create plot
plt.title("Learning Curve")
plt.xlabel("Training Set Size"), plt.ylabel("F1 Score"), plt.legend(loc="best")
plt.tight_layout()
plt.savefig('learning_curve_accuracy_svm_Game_F1.png')
plt.cla()

train_scores_val_game, test_scores_val_game = validation_curve(gridG.best_estimator_, XtrG, yTrG, param_name="max_iter",param_range=list(range(1,200,25)), n_jobs=-1, scoring='accuracy')
# Create means and standard deviations of training set scores
train_mean_val_game = np.mean(train_scores_val_game, axis=1)
train_std_val_game = np.std(train_scores_val_game, axis=1)
# Create means and standard deviations of test set scores
test_mean_val_game = np.mean(test_scores_val_game, axis=1)
test_std_val_game = np.std(test_scores_val_game, axis=1)
# Draw lines
plt.plot(list(range(1,200,25)), train_mean_val_game, '--', color="#111111",  label="Training score")
plt.plot(list(range(1,200,25)), test_mean_val_game, color="#111111", label="Testing score")
# Draw bands
plt.fill_between(list(range(1,200,25)), train_mean_val_game - train_std_val_game, train_mean_val_game + train_std_val_game, color="#DDDDDD")
plt.fill_between(list(range(1,200,25)), test_mean_val_game - test_std_val_game, test_mean_val_game + test_std_val_game, color="#DDDDDD")
# Create plot
plt.title("Iteration Learning Curve")
plt.xlabel("max iterations"), plt.ylabel("Accuracy"), plt.legend(loc="best")
plt.tight_layout()
plt.savefig('learning_curve_iteration_svm_game.png')
plt.cla()

train_scores_val_game_F1, test_scores_val_game_F1 = validation_curve(gridG.best_estimator_, XtrG, yTrG, param_name="max_iter",param_range=list(range(1,200,25)), n_jobs=-1, scoring='f1_weighted')
# Create means and standard deviations of training set scores
train_mean_val_game_F1 = np.mean(train_scores_val_game_F1, axis=1)
train_std_val_game_F1 = np.std(train_scores_val_game_F1, axis=1)
# Create means and standard deviations of test set scores
test_mean_val_game_F1 = np.mean(test_scores_val_game_F1, axis=1)
test_std_val_game_F1 = np.std(test_scores_val_game_F1, axis=1)
# Draw lines
plt.plot(list(range(1,200,25)), train_mean_val_game_F1, '--', color="#111111",  label="Training score")
plt.plot(list(range(1,200,25)), train_std_val_game_F1, color="#111111", label="Testing score")
# Draw bands
plt.fill_between(list(range(1,200,25)), train_mean_val_game_F1 - train_std_val_game_F1, train_mean_val_game_F1 + train_std_val_game_F1, color="#DDDDDD")
plt.fill_between(list(range(1,200,25)), test_mean_val_game_F1 - test_std_val_game_F1, test_mean_val_game_F1 + test_std_val_game_F1, color="#DDDDDD")
# Create plot
plt.title("Iteration Learning Curve")
plt.xlabel("max iterations"), plt.ylabel("F1 Score"), plt.legend(loc="best")
plt.tight_layout()
plt.savefig('learning_curve_iteration_svm_game_F1.png')
plt.cla()

scoresG = [x for x in gridG.cv_results_]
with open('scoresG.txt', 'w') as filehandle:
    for listitem in scoresG:
        filehandle.write('%s\n' % listitem)
with open('scoresG_F1.txt', 'w') as filehandle:
    for listitem in test_scoresG_f1:
        filehandle.write('%s\n' % listitem)


### CARS
start = time.time()
# defining parameter range 
param_grid = {'C': [1,3,5,10],  
              'gamma': [0.1, 0,3, 0.5, 0.6] ,
              'kernel': ['linear','poly']}  
              #'kernel': ['rbf', 'linear']}  
  
grid = GridSearchCV(SVC(random_state=42), param_grid, refit = True, verbose = 3) 
  
# fitting the model for grid search 
grid.fit(XtrC, yTrC) 

train_sizes, train_scores, test_scores = learning_curve(grid.best_estimator_, XtrC, yTrC, scoring='f1_weighted', n_jobs=-1, train_sizes=np.linspace(0.01, 1.0, 100))
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
plt.xlabel("Training Set Size"), plt.ylabel("F1 Score"), plt.legend(loc="best")
plt.tight_layout()
plt.savefig('learning_curve_accuracy_svm_f1.png')
plt.cla()
stop = time.time()
total_time = stop - start
print(total_time)
# print best parameter after tuning 
print(grid.best_params_) 
# print how our model looks after hyper-parameter tuning 
print(grid.best_estimator_) 
grid_predictions = grid.predict(XteC) 
# print classification report 
print(classification_report(yTeC, grid_predictions)) 

train_scores_val, test_scores_val = validation_curve(grid.best_estimator_, XtrC, yTrC, param_name="max_iter",param_range=list(range(1,200,25)), n_jobs=-1, scoring='accuracy')
# Create means and standard deviations of training set scores
train_mean_val = np.mean(train_scores_val, axis=1)
train_std_val = np.std(train_scores_val, axis=1)
# Create means and standard deviations of test set scores
test_mean_val = np.mean(test_scores_val, axis=1)
test_std_val = np.std(test_scores_val, axis=1)
# Draw lines
plt.plot(list(range(1,200,25)), train_mean_val, '--', color="#111111",  label="Training score")
plt.plot(list(range(1,200,25)), test_mean_val, color="#111111", label="Testing score")
# Draw bands
plt.fill_between(list(range(1,200,25)), train_mean_val - train_std_val, train_mean_val + train_std_val, color="#DDDDDD")
plt.fill_between(list(range(1,200,25)), test_mean_val - test_std_val, test_mean_val + test_std_val, color="#DDDDDD")
# Create plot
plt.title("Iteration Learning Curve")
plt.xlabel("max iterations"), plt.ylabel("Accuracy"), plt.legend(loc="best")
plt.tight_layout()
plt.savefig('learning_curve_iteration_svm.png')
plt.cla()

train_scores_val_F1, test_scores_val_F1 = validation_curve(grid.best_estimator_, XtrC, yTrC, param_name="max_iter",param_range=list(range(1,200,25)), n_jobs=-1, scoring='f1_weighted')
# Create means and standard deviations of training set scores
train_mean_val_F1 = np.mean(train_scores_val_F1, axis=1)
train_std_val_F1 = np.std(train_scores_val_F1, axis=1)
# Create means and standard deviations of test set scores
test_mean_val_F1 = np.mean(test_scores_val_F1, axis=1)
test_std_val_F1 = np.std(test_scores_val_F1, axis=1)
# Draw lines
plt.plot(list(range(1,200,25)), train_mean_val_F1, '--', color="#111111",  label="Training score")
plt.plot(list(range(1,200,25)), test_mean_val_F1, color="#111111", label="Testing score")
# Draw bands
plt.fill_between(list(range(1,200,25)), train_mean_val_F1 - train_std_val_F1, train_mean_val_F1 + train_std_val_F1, color="#DDDDDD")
plt.fill_between(list(range(1,200,25)), test_mean_val_F1 - test_std_val_F1, test_mean_val_F1 + test_std_val_F1, color="#DDDDDD")
# Create plot
plt.title("Iteration Learning Curve")
plt.xlabel("max iterations"), plt.ylabel("F1 Score"), plt.legend(loc="best")
plt.tight_layout()
plt.savefig('learning_curve_iteration_svm_F1.png')
plt.cla()

scores = [x[1] for x in grid.cv_results_]
with open('scores.txt', 'w') as filehandle:
    for listitem in scores:
        filehandle.write('%s\n' % listitem)
        
scores0 = [x[0] for x in grid.cv_results_]
with open('scores0.txt', 'w') as filehandle:
    for listitem in scores0:
        filehandle.write('%s\n' % listitem)

with open('scoresf1.txt', 'w') as filehandle:
    for listitem in test_scores:
        filehandle.write('%s\n' % listitem)

