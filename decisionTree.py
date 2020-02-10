from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import learning_curve, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import matplotlib.pyplot as plt
import time 
from processdata import XtrC, XteC, yTrC, yTeC, XtrG, XteG, yTrG, yTeG
plt.cla()

#cars classifier
start = time.time()
dt = DecisionTreeClassifier(random_state=42)
#pruning
path = dt.cost_complexity_pruning_path(XtrC, yTrC)
ccp_alphas, impurities = path.ccp_alphas, path.impurities
# gridCV
'''params = {'ccp_alpha': ccp_alphas}
clf = GridSearchCV(dt, params)
clf.fit(XtrC, yTrC)
train_prediction = clf.predict(XtrC)
test_prediction = clf.predict(XteC)
train_classification_report = classification_report(yTrC, train_prediction)
test_classification_report = classification_report(yTeC, test_prediction)
print(train_classification_report)
print(test_classification_report)
print(clf.best_score_)
print(clf.best_params_)
train_sizes, train_scores, test_scores = learning_curve(clf.best_estimator_, XtrC, yTrC, n_jobs=-1, scoring='accuracy', train_sizes=np.linspace(0.01, 1.0, 100))
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
plt.savefig('learning_curve_accuracy.png')
plt.cla()'''

# iteration
clfs = []
for ccp_alpha in ccp_alphas:
    clf = DecisionTreeClassifier(random_state=42, ccp_alpha=ccp_alpha)
    clf.fit(XtrC, yTrC)
    clfs.append(clf)

clfs = clfs[:-1]
ccp_alphas = ccp_alphas[:-1]
training_scores = [clf.score(XtrC, yTrC) for clf in clfs]
training_scores_f1 = [f1_score(yTrC, clf.predict(XtrC), average='weighted') for clf in clfs]
training_score_clf = [{clf: clf.score(XtrC, yTrC)} for clf in clfs]
#print(training_score_clf)
fig, ax = plt.subplots()
ax.set_xlabel("alpha")
ax.set_ylabel("accuracy")
ax.set_title("Validation Curve")
ax.plot(ccp_alphas, training_scores, marker='o', label="train",
        drawstyle="steps-post")
ax.legend()
plt.savefig('training_tuning_iteration.png')
plt.cla()

#f1
# train_sizes, train_scores, test_scores = learning_curve(DecisionTreeClassifier(random_state=42, ccp_alpha=0.04), XtrC, yTrC, n_jobs=-1, scoring='accuracy', train_sizes=np.linspace(0.01, 1.0, 100))
train_sizes, train_scores, test_scores = learning_curve(DecisionTreeClassifier(random_state=42, ccp_alpha=0.04), XtrC, yTrC, n_jobs=-1, scoring='f1_weighted', train_sizes=np.linspace(0.01, 1.0, 100))

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
# plt.xlabel("Training Set Size"), plt.ylabel("Accuracy"), plt.legend(loc="best")
# f1
plt.xlabel("Training Set Size"), plt.ylabel("F1 Score"), plt.legend(loc="best")
plt.tight_layout()
# plt.savefig('learning_curve_accuracy.png')
# f1
plt.savefig('learning_curve_accuracy_decision_f1.png')
plt.cla()
stop = time.time()
total_time = stop - start
print(total_time)

prClf = DecisionTreeClassifier(random_state=42, ccp_alpha=0.04)
prClf.fit(XtrC, yTrC)
y_pred = prClf.predict(XteC)
print(classification_report(yTeC, y_pred))
f1 = f1_score(yTeC, y_pred, average='weighted')
print(f1)

testing_scores = [clf.score(XteC, yTeC) for clf in clfs]
testing_scores_f1 = [f1_score(yTeC, clf.predict(XteC), average='weighted') for clf in clfs]

#cars validation curve
fig, ax = plt.subplots()
ax.set_xlabel("alpha")
ax.set_ylabel("accuracy")
ax.set_title("Validation Curve")
'''ax.plot(ccp_alphas, training_scores, marker='o', label="train",
        drawstyle="steps-post")'''
#f1
ax.plot(ccp_alphas, training_scores_f1, marker='o', label="train",
        drawstyle="steps-post")
'''ax.plot(ccp_alphas, testing_scores, marker='o', label="test",
        drawstyle="steps-post")'''
#f1
ax.plot(ccp_alphas, testing_scores_f1, marker='o', label="test",
        drawstyle="steps-post")
ax.legend()
plt.savefig('validation_curve_f1_decision.png')
plt.cla()


######## game
start_game = time.time()
dtG = DecisionTreeClassifier(random_state=42)
pathG = dtG.cost_complexity_pruning_path(XtrG, yTrG)
ccp_alphasG, impuritiesG = pathG.ccp_alphas, pathG.impurities

'''# gridCV
params_G = {'ccp_alpha': ccp_alphasG}
clfG = GridSearchCV(dtG, params_G)
clfG.fit(XtrG, yTrG)
train_predictionG = clfG.predict(XtrG)
test_predictionG = clfG.predict(XteG)
train_classification_reportG = classification_report(yTrG, train_predictionG)
test_classification_reportG = classification_report(yTeG, test_predictionG)
print(train_classification_reportG)
print(test_classification_reportG)
print(clfG.best_score_)
print(clfG.best_params_)
train_sizesG, train_scoresG, test_scoresG = learning_curve(clfG.best_estimator_, XtrG, yTrG, n_jobs=-1, scoring='accuracy', train_sizes=np.linspace(0.01, 1.0, 100))
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
plt.savefig('learning_curve_accuracy_game.png')
plt.cla()'''

# iteration
clfGs = []
for ccp_alpha in ccp_alphasG:
    clfG = DecisionTreeClassifier(random_state=42, ccp_alpha=ccp_alpha)
    clfG.fit(XtrG, yTrG)
    clfGs.append(clfG)

clfGs = clfGs[:-1]
ccp_alphasG = ccp_alphasG[:-1]

training_scoresG = [clf.score(XtrG, yTrG) for clf in clfGs]
training_scoresG_f1 = [f1_score(yTrG, clf.predict(XtrG), average='weighted') for clf in clfGs]

training_score_clfG = [{clf: clf.score(XtrG, yTrG)} for clf in clfGs]
#print(training_score_clfG)
fig, ax = plt.subplots()
ax.set_xlabel("alpha")
ax.set_ylabel("accuracy")
ax.set_title("Validation Curve")
ax.plot(ccp_alphasG, training_scoresG, marker='o', label="train",
        drawstyle="steps-post")
ax.legend()
plt.savefig('training_tuning_iteration_game.png')
plt.cla()

#train_sizesG, train_scoresG, test_scoresG = learning_curve(DecisionTreeClassifier(random_state=42, ccp_alpha=0.016), XtrG, yTrG, n_jobs=-1, scoring='accuracy', train_sizes=np.linspace(0.01, 1.0, 100))
# f1
train_sizesG, train_scoresG, test_scoresG = learning_curve(DecisionTreeClassifier(random_state=42, ccp_alpha=0.016), XtrG, yTrG, n_jobs=-1, scoring='f1_weighted', train_sizes=np.linspace(0.01, 1.0, 100))

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
#plt.xlabel("Training Set Size"), plt.ylabel("Accuracy"), plt.legend(loc="best")
#f1
plt.xlabel("Training Set Size"), plt.ylabel("F1 Score"), plt.legend(loc="best")

plt.tight_layout()
plt.savefig('learning_curve_accuracy_game_iteration_F1.png')
plt.cla()
stop_game = time.time()
total_time_game = stop_game - start_game
print(total_time_game)

prClfG = DecisionTreeClassifier(random_state=42, ccp_alpha=0.016)
prClfG.fit(XtrG, yTrG)
y_predG = prClfG.predict(XteG)
print(classification_report(yTeG, y_predG))
f1G = f1_score(yTeG, y_predG, average='weighted')
print(f1G)

testing_scoresG = [clf.score(XteG, yTeG) for clf in clfGs]
#f1
testing_scoresG_f1 = [f1_score(yTeG, clf.predict(XteG), average='weighted') for clf in clfGs]
#cars validation curve
fig, ax = plt.subplots()
ax.set_xlabel("alpha")
ax.set_ylabel("F1 Score")
ax.set_title("Validation Curve")
'''ax.plot(ccp_alphasG, training_scoresG, marker='o', label="train",
        drawstyle="steps-post")'''
'''ax.plot(ccp_alphasG, testing_scoresG, marker='o', label="test",
        drawstyle="steps-post")'''
#f1
ax.plot(ccp_alphasG, training_scoresG_f1, marker='o', label="train",
        drawstyle="steps-post")
#f1
ax.plot(ccp_alphasG, testing_scoresG_f1, marker='o', label="test",
        drawstyle="steps-post")
ax.legend()
# plt.savefig('validation_curve_game_decision.png')
#f1
plt.savefig('validation_curve_game_F1_decision.png')
plt.cla()
