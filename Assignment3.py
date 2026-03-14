#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline


import seaborn as sns
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler

from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression



#%%
# read the data file into dataframes and merge into one single dataframe for easier analysis
os.chdir("C:\\Users\\vikto\\Desktop\\Skola'\\Andra Kurser\\D7015B - Industrial AI and eMaintenence\\Assignment 3")
trail1 = pd.read_csv("trail1_data.csv")
trail2 = pd.read_csv("trail2_data.csv")
trail3 = pd.read_csv("trail3_data.csv")

#%%
#Merge the three trails into one DataFrame for easier analysis
merged_data = pd.concat([trail1, trail2, trail3], ignore_index=True)
print(merged_data.shape)
merged_data.drop(columns = ['start_time', 'axle', 'cluster', 'tsne_1', 'tsne_2'], inplace = True) #Drop the columns that are not needed for the analysis
merged_data["event"] = (merged_data["event"].str.lower() != "normal").astype(int) #Convert the event column to binary, where 0 = normal event and 1 is for any other event
print(merged_data.shape)

print(merged_data) #Verify data and check structure

#%% Split up columns so we train on features (X) to predict (Y - the event column)

#When normalizing the data, we want to exclude the event column since it is is a binary variable and we should only normalize features (X).
#We will normalize only on the trained data (X_train) and then apply it to test_data (X_test) to not let the model have access to test data during training. 
#This is important to prevent data leakage and ensure that the model generalizes well to unseen data.
X = merged_data.drop(columns=['event']) 
print("Number of features:", X.shape[1])
y = merged_data['event']

#%% 80-20 train-test split (20 % for test, 80 % for training)
X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, #y is the target variable (event) for training and testing.
                                                    test_size=0.2, #20 % of data for test and 80 % for training
                                                    random_state=42, 
                                                    stratify=y) #Same proportion of normal/abnormal events in both training and test sets

#%% Normalization of data and training of SVM model

#We want to normalize our data before analyzing it. This is to balance scaling between different feautures that might have different ranges and units.
scaler = StandardScaler() #Standardscaler comes from sklearn.preproccing package and standardizes to N(0,1) distiribution
X_train_scaled = scaler.fit_transform(X_train) #Fit the scaler on the training data and transform it
X_test_scaled = scaler.transform(X_test) #Transform the test data using the same scaler fitted on the training data

svm_model = SVC(kernel='rbf', probability=True, random_state=42) #Create an SVM model with RBF kernel, random_state=42 basically means we can re-use the same seed
svm_model.fit(X_train_scaled, y_train) #Fit the SVM model on the scaled training data (80/20 split)
#%% Evaluation of the model

#Evaluate the model on the test set
y_pred = svm_model.predict(X_test_scaled) #Predict the labels for the test set using the trained SVM model

print(classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=svm_model.classes_) 
disp.plot() #Plot the confusion matrix
plt.title("Confusion Matrix for 80-20 train split (0 = Normal Event, 1 = Abnormal Event)")
plt.show()
#%% 5-fold cross-validation
    
#5-fold cross-validation splits training data into 5 subsets, trains on 4 of these and test on remaining subsets, and repeats for each subset. This model is then used on test set to evaluate performance.
pipeline = make_pipeline(StandardScaler(), SVC(kernel='rbf', probability=True, random_state=42)) #Create a pipeline that includes standard scaling and new SVM model to train on. 
cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5)
print("CV scores:", cv_scores)
print("Mean CV score:", np.mean(cv_scores))

#Hyperparameter grid for RBF SVM
param_grid = { #Parameters to search over for tuning the SVM model. Total 5x5 = 25 combinations to evaluate over cv = 5 folds --> 25*5 = 125 total model fits to evaluate and compare.
    'svc__C': [0.1, 0.05, 1, 10, 50], #C regulates trade-off between training error and testing error. High C prioritize minimizing training errors (sensitive to outliers). Low C allows more training errors but can improve generalization.
    'svc__gamma': [0.01, 0.0625, 0.1, 0.5, 1] #Gamma describes how local training points influence other data points. High gamma means data points only affect nearby points, lower gamma means data points have further reach and thus higher weight on more distant points.
}

grid = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1) #Creates the searchgrid
grid.fit(X_train, y_train) #Fit grid to training data, this will evaluate all combinations of hyperparameters in param_grid using 5-fold cross-validation and find the best combination based on mean CV score.
grid.cv_results_
print("Best CV score:", grid.best_score_) #Best mean CV score across all hyperparameter combinations
#Best hyperparameters from CV
print("Best hyperparameters from 5-fold CV:", grid.best_params_)

#Evaluate final CV-tuned model on test set
best_model = grid.best_estimator_
y_test_pred_cv = best_model.predict(X_test)

print("CV-Tuned SVM Performance on Test Set:")
print(classification_report(y_test, y_test_pred_cv)) 

cm_cv = confusion_matrix(y_test, y_test_pred_cv) #Compute the confusion matrix for trained cv model on test set
disp_cv = ConfusionMatrixDisplay(confusion_matrix=cm_cv, display_labels=best_model.named_steps['svc'].classes_)
disp_cv.plot()
plt.title("Confusion Matrix for CV-trained SVM (0 = Normal, 1 = Abnormal)")
plt.show()

##FOR GRADE 5

#Separate features and target
X = merged_data.drop(columns=['event'])
y = merged_data['event']

#%%
#Full Pearson correlation matrix
corr_matrix = merged_data.corr()

plt.figure(figsize=(12,10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title("Full Pearson Correlation Matrix")
plt.show()

#%%
#Mutual Information scores for all features
mi_scores = mutual_info_classif(X, y, random_state=42)
mi_df = pd.DataFrame({'feature': X.columns, 'MI_score': mi_scores}).sort_values(by='MI_score', ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(x='MI_score', y='feature', data=mi_df)
plt.title("Full Mutual Information Scores")
plt.show()

#%%
#Recursive Feature Elimination (RFE):

svc_estimator = SVC(kernel='linear', random_state=42)  #Using a linear kernel for RFE to get feature importance based on weights. Using a linear kernel might miss non-linear relationships.
rfe = RFE(estimator=svc_estimator, n_features_to_select=6) #Select top 6 features based on RFE
pipeline_rfe = make_pipeline(StandardScaler(), rfe, svc_estimator)
pipeline_rfe.fit(X, y)
selected_features_rfe = X.columns[rfe.support_].tolist()
print("RFE selected features:", selected_features_rfe)

#%%

pipeline_logregL1 = make_pipeline(
    StandardScaler(),
    LogisticRegression(penalty='l1', solver='liblinear', C = 1.0,random_state=42) #C = 1/alpha, where alpha is penalty strength (regularization parameter). Read more about this in research section of report.
)
pipeline_logregL1.fit(X, y)

coef = pipeline_logregL1.named_steps['logisticregression'].coef_[0]
selected_features_l1 = X.columns[coef != 0].tolist()

print("\nLASSO selected features:", selected_features_l1)


#%%
filter_features = ['kurtosis', 'max', 'zero_crossings', 'spectral_flatness', 'skewness', 'spectral_bandwidth']
print("Filter selected features:", filter_features) #To visualize the filter selected features in terminal (was selected manually based on mutual inormation scores and pearson correleation)
rfe_features = selected_features_rfe
lasso_features = selected_features_l1

X_filter = X[filter_features]
X_rfe = X[rfe_features]
X_lasso = X[lasso_features]

def evaluate_features(X_sub, y):
    pipeline = make_pipeline(StandardScaler(), SVC(kernel='rbf', probability=True, random_state=42))
    scores = cross_val_score(pipeline, X_sub, y, cv=5)
    mean_score = float(np.mean(scores))
    return mean_score


Filter_Score = evaluate_features(X_filter, y)
print("Filter features performance:", Filter_Score)

RFE_Score = evaluate_features(X_rfe, y)
print("RFE features performance:", RFE_Score)

LASSO_Score = evaluate_features(X_lasso, y)
print("LASSO features performance:", LASSO_Score)

