import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV, train_test_split, cross_validate, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score, roc_curve
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
from skopt.plots import plot_convergence
os.chdir('C:/Users/23985/Desktop/实验测试/datadata/数据/Magainin2/data1')
train0 = pd.read_csv('Magainin2_1252_guiyihua_data1_0.csv') 
train1 = pd.read_csv('Magainin2_1252_guiyihua_data1_30.csv')
train = pd.concat([train0, train1], axis=0)
os.environ['PYDEVD_DISABLE_FILE_VALIDATION'] = '1'
train['sample'] = train['sample'].str.split(' ').str[0].str.split('#').str[1].astype(int)
ax = train.groupby('sample').mean().T.plot(kind='line', figsize=(15, 6))
raman_shift_train = train.columns[1:].to_numpy().astype(int)
condition_train = np.concatenate([
    [True],  
    ((raman_shift_train >= 700) & (raman_shift_train <= 1800)) |
    ((raman_shift_train >= 2750) & (raman_shift_train <= 2950))
])
train = train.iloc[:, condition_train]
ax = train.groupby('sample').mean().T.plot(kind='line', figsize=(15, 6))
train.iloc[[1, 10, 30, 70, 300, 405],:].set_index('sample').T.plot(kind='line', figsize=(15, 5), color = ['blue', 'blue', 'blue', 'red', 'red', 'red'])
X = train.drop('sample', axis=1).fillna(0)
Y = train['sample']
trainX, testX, trainY, testY= train_test_split(X, Y, random_state=10, train_size=0.8, shuffle=True, stratify=Y)
# XGBoost
model=XGBClassifier(eval_metric='logloss')
param_grid={   'n_estimators': [10],
               'max_depth': [10],
              'learning_rate': [0.1, 0.5],
               'reg_alpha': [0.1, 0.5],
               'reg_lambda': [0.1, 1]
}
grid_search = GridSearchCV(model,param_grid,cv=3,verbose=1,n_jobs=-1,scoring='accuracy')
grid_search.fit(trainX, trainY.ravel())  
best_model = grid_search.best_estimator_ 
best_params = grid_search.best_params_
write_module_path = 'C:/Users/23985/Desktop/data/XGBoost/output/'
if not os.path.exists(write_module_path):
    os.makedirs(write_module_path)
best_model.save_model(os.path.join(write_module_path, 'XGBoost.cbm'))
print({'best_params：': best_params})
from sklearn.metrics import accuracy_score
predictions=best_model.predict(testX)
accuracy = accuracy_score(testY, predictions) 
accuracy_train = accuracy_score(trainY, best_model.predict(trainX))  
print({'accuracy_test': f"{accuracy*100}%"})    
print({'accuracy_train': f"{accuracy_train*100}%"}) 
y_scores = best_model.predict_proba(testX)[:, 1]
print({'accuracy_predictions': f"{accuracy*100}%"}) 
cm = confusion_matrix(testY, predictions)
plt.figure(figsize=(10,7)) 
ax = sns.heatmap(cm, annot=True, fmt='d', annot_kws={"size": 26, "fontname": "Arial"},
                 cmap='Reds', square=True)
plt.xlabel('Predicted Label', fontsize=26, fontname="Arial")
plt.ylabel('True Label', fontsize=26, fontname="Arial")
plt.xticks(fontsize=26, fontname="Arial")
plt.yticks(fontsize=26, fontname="Arial")
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=26) 
cbar.ax.set_yticklabels(cbar.ax.get_yticklabels(), fontname="Arial")  
plt.show()
y_scores = best_model.predict_proba(testX)[:, 1]
fpr, tpr, thresholds = roc_curve(testY, y_scores)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='#800000', lw=2, label='ROC curve (area = %0.2f )' % roc_auc)
plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
plt.xlabel('False positive rate', fontsize=26, fontname='Arial')
plt.ylabel('True positive rate', fontsize=26, fontname='Arial')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xticks(fontsize=26, fontname='Arial')
plt.yticks(fontsize=26, fontname='Arial')
plt.text(0.6, 0.1, f'AUC = {roc_auc:.2f}', fontsize=26, fontname='Arial', color='#800000')
plt.show()
roc_data = pd.DataFrame({
    'False Positive rate': fpr,
    'True Positive rate': tpr,
    'Thresholds': thresholds
})
roc_data.to_csv('roc_curve_XGBoost_magainin2.csv', index=False)
df = pd.concat([pd.Series(testX.columns), pd.Series(best_model.feature_importances_)], axis=1)\
    .sort_values(by=1, ascending=False) 
df.columns = ['Raman shift', 'Importance']
df_top30 = df.iloc[:10, :]
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 26
df_top30.plot(kind='bar', x='Raman shift', y='Importance', color='#800000', figsize=(24, 7))
plt.xlabel('Raman shift')
plt.ylabel('VarImp')
plt.legend(['Importance'])
plt.show()