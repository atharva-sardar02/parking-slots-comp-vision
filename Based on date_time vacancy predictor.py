# Import necessary libraries
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import numpy as np
import warnings
# from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, auc

warnings.filterwarnings("ignore")


# def plot_multiclass_roc_curve(classifier, X_data, y_data):
#     y_scores = classifier.predict_proba(X_data)
#     plt.figure(figsize=(8, 8))
#     for i in range(len(classifier.classes_)):
#         fpr, tpr, _ = roc_curve(y_data == classifier.classes_[i], y_scores[:, i])
#         roc_auc = auc(fpr, tpr)
#         plt.plot(fpr, tpr, label=f'Class {classifier.classes_[i]} (AUC = {roc_auc:.2f})')
#     plt.plot([0, 1], [0, 1], color='navy', linestyle='--', linewidth=2, label='Random', alpha=0.8)
#     plt.xlim([0.0, 1.0])
#     plt.ylim([0.0, 1.05])
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title('Receiver Operating Characteristic (ROC) Curve (Multiclass)')
#     plt.legend(loc='lower right')
#     plt.show()


# # Sample data (replace this with your actual dataset)
# data = {'Date': ['2023-01-03', '2023-01-03', '2023-01-04', '2023-01-04', '2023-01-05',
#                  '2023-01-05', '2023-01-06', '2023-01-06', '2023-01-07', '2023-01-07'],
#         'Day': ['Wednesday', 'Wednesday', 'Thursday', 'Thursday', 'Friday',
#                 'Friday', 'Saturday', 'Saturday', 'Sunday', 'Sunday'],
#         'Time': ['09:00', '15:00', '11:00', '13:00', '10:30',
#                  '12:30', '14:30', '16:30', '18:00', '20:00'],
#         'Vacant': ['No', 'Yes', 'Yes', 'No', 'No',
#                    'Yes', 'Yes', 'No', 'Yes', 'No']}

# Generate dummy data with k entries
k = 1000
np.random.seed(5805)
date_range = pd.date_range(start="2023-01-01", periods=k, freq="D")
day_choices = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
days = np.random.choice(day_choices, size=k)
times = pd.to_datetime(np.random.choice(pd.date_range(start="08:00", end="20:00", freq="H"), size=k), format='%H:%M')
vacant_choices = ['Yes', 'No']
vacant = np.random.choice(vacant_choices, size=k)

data = {'Date': date_range, 'Day': days, 'Time': times, 'Vacant': vacant}
df = pd.DataFrame(data)

# Convert 'Date' to datetime objects
df['Date'] = pd.to_datetime(df['Date'])
# Convert 'Date' to numerical representations
df['Date'] = df['Date'].dt.dayofyear

# Convert 'Time' to datetime objects
df['Time'] = pd.to_datetime(df['Time'], format='%H:%M')
# Convert 'Time' to minutes
df['Time'] = df['Time'].apply(lambda x: x.hour * 60 + x.minute)

# Convert categorical variables to numerical using Label Encoding
le_day = LabelEncoder()
df['Day'] = le_day.fit_transform(df['Day'])
le_vacant = LabelEncoder()
df['Vacant'] = le_vacant.fit_transform(df['Vacant'])

# Separate features (X) and target variable (y)
X = df[['Date', 'Day', 'Time']]
y = df['Vacant']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5805)

classification_table = pd.DataFrame(columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'Specificity', 'Cross Validation Mean Score'])

# Initialize the Decision Tree Classifier
dt_clf = DecisionTreeClassifier()
# Perform cross-validation
cv_scores = cross_val_score(dt_clf, X, y, cv=5)
print("Decision Tree Cross-validation scores:", cv_scores)
print("Decision Tree Mean accuracy: {:.2f}".format(cv_scores.mean()))
# Fit the model on the training data
dt_clf.fit(X_train, y_train)
# Make predictions on the test data
y_pred = dt_clf.predict(X_test)
# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Decision Tree Accuracy: {accuracy:.2f}')
# Display classification report
print('Decision Tree Classification Report:')
print(classification_report(y_test, y_pred))
# Visualize the decision tree using matplotlib
plt.figure(figsize=(12, 8))
plot_tree(dt_clf, filled=True, feature_names=X.columns, class_names=[str(label) for label in le_vacant.classes_],
          rounded=True, proportion=True)
plt.show()
# plot_multiclass_roc_curve(dt_clf, X_test, y_test)
# scores = cross_val_score(dt_clf, X_train, y_train, cv=5, scoring='accuracy')
# classification_table = pd.concat([classification_table, pd.DataFrame({'Model': 'Decision Tree',
#                                 'Accuracy': accuracy_score(y_test, y_pred).round(2),
#                                 'Precision': precision_score(y_test, y_pred, average='weighted').round(2),
#                                 'Recall': recall_score(y_test, y_pred, average='weighted').round(2),
#                                 'F1 Score': f1_score(y_test, y_pred, average='weighted').round(2),
#                                 'Specificity': recall_score(y_test, y_pred, pos_label=0, average='weighted').round(2),
#                                 'Cross Validation Mean Score': scores.mean().round(2)}, index=[0])], axis=0)

# Initialize the Neural Network Classifier
nn_clf = MLPClassifier(random_state=5805)
# Perform cross-validation for Neural Network
cv_scores_nn = cross_val_score(nn_clf, X, y, cv=5)
print("Neural Network Cross-validation scores:", cv_scores_nn)
print("Neural Network Mean accuracy: {:.2f}".format(cv_scores_nn.mean()))
# Fit the Neural Network model on the training data
nn_clf.fit(X_train, y_train)
# Make predictions on the test data using Neural Network
y_pred_nn = nn_clf.predict(X_test)
# Evaluate the Neural Network model
accuracy_nn = accuracy_score(y_test, y_pred_nn)
print(f'Neural Network Accuracy: {accuracy_nn:.2f}')
# Display Neural Network classification report
print('Neural Network Classification Report:')
print(classification_report(y_test, y_pred_nn))
# plot_multiclass_roc_curve(nn_clf, X_test, y_test)
# scores = cross_val_score(nn_clf, X_train, y_train, cv=5, scoring='accuracy')
# classification_table = pd.concat([classification_table, pd.DataFrame({'Model': 'Neural Network(MLP)',
#                                 'Accuracy': accuracy_score(y_test, y_pred).round(2),
#                                 'Precision': precision_score(y_test, y_pred, average='weighted').round(2),
#                                 'Recall': recall_score(y_test, y_pred, average='weighted').round(2),
#                                 'F1 Score': f1_score(y_test, y_pred, average='weighted').round(2),
#                                 'Specificity': recall_score(y_test, y_pred, pos_label=0, average='weighted').round(2),
#                                 'Cross Validation Mean Score': scores.mean().round(2)}, index=[0])], axis=0)

# Initialize the Random Forest Classifier
rf_clf = RandomForestClassifier(random_state=5805)
# Perform cross-validation for Random Forest
cv_scores_rf = cross_val_score(rf_clf, X, y, cv=5)
print("Random Forest Cross-validation scores:", cv_scores_rf)
print("Random Forest Mean accuracy: {:.2f}".format(cv_scores_rf.mean()))
# Fit the Random Forest model on the training data
rf_clf.fit(X_train, y_train)
# Make predictions on the test data using Random Forest
y_pred_rf = rf_clf.predict(X_test)
# Evaluate the Random Forest model
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f'Random Forest Accuracy: {accuracy_rf:.2f}')
# Display Random Forest classification report
print('Random Forest Classification Report:')
print(classification_report(y_test, y_pred_rf))
# plot_multiclass_roc_curve(rf_clf, X_test, y_test)
# scores = cross_val_score(rf_clf, X_train, y_train, cv=5, scoring='accuracy')
# classification_table = pd.concat([classification_table, pd.DataFrame({'Model': 'Random Forest',
#                                 'Accuracy': accuracy_score(y_test, y_pred).round(2),
#                                 'Precision': precision_score(y_test, y_pred, average='weighted').round(2),
#                                 'Recall': recall_score(y_test, y_pred, average='weighted').round(2),
#                                 'F1 Score': f1_score(y_test, y_pred, average='weighted').round(2),
#                                 'Specificity': recall_score(y_test, y_pred, pos_label=0, average='weighted').round(2),
#                                 'Cross Validation Mean Score': scores.mean().round(2)}, index=[0])], axis=0)

# Initialize the SVM Classifier
svm_clf = SVC(random_state=5805)
# Perform cross-validation for SVM
cv_scores_svm = cross_val_score(svm_clf, X, y, cv=5)
print("SVM Cross-validation scores:", cv_scores_svm)
print("SVM Mean accuracy: {:.2f}".format(cv_scores_svm.mean()))
# Fit the SVM model on the training data
svm_clf.fit(X_train, y_train)
# Make predictions on the test data using SVM
y_pred_svm = svm_clf.predict(X_test)
# Evaluate the SVM model
accuracy_svm = accuracy_score(y_test, y_pred_svm)
print(f'SVM Accuracy: {accuracy_svm:.2f}')
# Display SVM classification report
print('SVM Classification Report:')
print(classification_report(y_test, y_pred_svm))
# plot_multiclass_roc_curve(svm_clf, X_test, y_test)
# scores = cross_val_score(svm_clf, X_train, y_train, cv=5, scoring='accuracy')
# classification_table = pd.concat([classification_table, pd.DataFrame({'Model': 'SVM',
#                                 'Accuracy': accuracy_score(y_test, y_pred).round(2),
#                                 'Precision': precision_score(y_test, y_pred, average='weighted').round(2),
#                                 'Recall': recall_score(y_test, y_pred, average='weighted').round(2),
#                                 'F1 Score': f1_score(y_test, y_pred, average='weighted').round(2),
#                                 'Specificity': recall_score(y_test, y_pred, pos_label=0, average='weighted').round(2),
#                                 'Cross Validation Mean Score': scores.mean().round(2)}, index=[0])], axis=0)

# Initialize the Logistic Regression Classifier
lr_clf = LogisticRegression(random_state=5805)
# Perform cross-validation for Logistic Regression
cv_scores_lr = cross_val_score(lr_clf, X, y, cv=5)
print("Logistic Regression Cross-validation scores:", cv_scores_lr)
print("Logistic Regression Mean accuracy: {:.2f}".format(cv_scores_lr.mean()))
# Fit the Logistic Regression model on the training data
lr_clf.fit(X_train, y_train)
# Make predictions on the test data using Logistic Regression
y_pred_lr = lr_clf.predict(X_test)
# Evaluate the Logistic Regression model
accuracy_lr = accuracy_score(y_test, y_pred_lr)
print(f'Logistic Regression Accuracy: {accuracy_lr:.2f}')
# Display Logistic Regression classification report
print('Logistic Regression Classification Report:')
print(classification_report(y_test, y_pred_lr))
# plot_multiclass_roc_curve(lr_clf, X_test, y_test)
# scores = cross_val_score(lr_clf, X_train, y_train, cv=5, scoring='accuracy')
# classification_table = pd.concat([classification_table, pd.DataFrame({'Model': 'Logistic Regression',
#                                 'Accuracy': accuracy_score(y_test, y_pred).round(2),
#                                 'Precision': precision_score(y_test, y_pred, average='weighted').round(2),
#                                 'Recall': recall_score(y_test, y_pred, average='weighted').round(2),
#                                 'F1 Score': f1_score(y_test, y_pred, average='weighted').round(2),
#                                 'Specificity': recall_score(y_test, y_pred, pos_label=0, average='weighted').round(2),
#                                 'Cross Validation Mean Score': scores.mean().round(2)}, index=[0])], axis=0)

# Initialize the Naïve Bayes Classifier
nb_clf = GaussianNB()
# Perform cross-validation for Naïve Bayes
cv_scores_nb = cross_val_score(nb_clf, X, y, cv=5)
print("Naïve Bayes Cross-validation scores:", cv_scores_nb)
print("Naïve Bayes Mean accuracy: {:.2f}".format(cv_scores_nb.mean()))
# Fit the Naïve Bayes model on the training data
nb_clf.fit(X_train, y_train)
# Make predictions on the test data using Naïve Bayes
y_pred_nb = nb_clf.predict(X_test)
# Evaluate the Naïve Bayes model
accuracy_nb = accuracy_score(y_test, y_pred_nb)
print(f'Naïve Bayes Accuracy: {accuracy_nb:.2f}')
# Display Naïve Bayes classification report
print('Naïve Bayes Classification Report:')
print(classification_report(y_test, y_pred_nb))
# plot_multiclass_roc_curve(nn_clf, X_test, y_test)
# scores = cross_val_score(nb_clf, X_train, y_train, cv=5, scoring='accuracy')
# classification_table = pd.concat([classification_table, pd.DataFrame({'Model': 'Naive Bayes',
#                                 'Accuracy': accuracy_score(y_test, y_pred).round(2),
#                                 'Precision': precision_score(y_test, y_pred, average='weighted').round(2),
#                                 'Recall': recall_score(y_test, y_pred, average='weighted').round(2),
#                                 'F1 Score': f1_score(y_test, y_pred, average='weighted').round(2),
#                                 'Specificity': recall_score(y_test, y_pred, pos_label=0, average='weighted').round(2),
#                                 'Cross Validation Mean Score': scores.mean().round(2)}, index=[0])], axis=0)

# Initialize the KNN Classifier
knn_clf = KNeighborsClassifier()
# Perform cross-validation for KNN
cv_scores_knn = cross_val_score(knn_clf, X, y, cv=5)
print("KNN Cross-validation scores:", cv_scores_knn)
print("KNN Mean accuracy: {:.2f}".format(cv_scores_knn.mean()))
# Fit the KNN model on the training data
knn_clf.fit(X_train, y_train)
# Make predictions on the test data using KNN
y_pred_knn = knn_clf.predict(X_test)
# Evaluate the KNN model
accuracy_knn = accuracy_score(y_test, y_pred_knn)
print(f'KNN Accuracy: {accuracy_knn:.2f}')
# Display KNN classification report
print('KNN Classification Report:')
print(classification_report(y_test, y_pred_knn))
# plot_multiclass_roc_curve(knn_clf, X_test, y_test)
# scores = cross_val_score(knn_clf, X_train, y_train, cv=5, scoring='accuracy')
# classification_table = pd.concat([classification_table, pd.DataFrame({'Model': 'KNN',
#                                 'Accuracy': accuracy_score(y_test, y_pred).round(2),
#                                 'Precision': precision_score(y_test, y_pred, average='weighted').round(2),
#                                 'Recall': recall_score(y_test, y_pred, average='weighted').round(2),
#                                 'F1 Score': f1_score(y_test, y_pred, average='weighted').round(2),
#                                 'Specificity': recall_score(y_test, y_pred, pos_label=0, average='weighted').round(2),
#                                 'Cross Validation Mean Score': scores.mean().round(2)}, index=[0])], axis=0)
