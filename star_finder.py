import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Charger le dataset
file_path = 'dataset/star_classification.csv'
data = pd.read_csv(file_path)

# Exploration des données
print(data.head())
print(data.info())

# Vérification des valeurs manquantes
print(data.isnull().sum())

# Préparation des données
X = data.drop('class', axis=1)  # Supprimer la colonne cible
y = data['class']  # Colonne cible

# Normalisation des données
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Division en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# KNN Classifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

# Évaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Initialiser l'arbre de décision
decision_tree = DecisionTreeClassifier(random_state=42)

# Entraîner le modèle
decision_tree.fit(X_train, y_train)

# Prédire les classes
y_pred_dt = decision_tree.predict(X_test)

# Évaluation
print("Arbre de Décision - Accuracy:", accuracy_score(y_test, y_pred_dt))
print("Classification Report:\n", classification_report(y_test, y_pred_dt))

# Initialiser le classifieur bayésien
naive_bayes = GaussianNB()

# Entraîner le modèle
naive_bayes.fit(X_train, y_train)

# Prédire les classes
y_pred_nb = naive_bayes.predict(X_test)

# Évaluation
print("Classifieur Bayésien - Accuracy:", accuracy_score(y_test, y_pred_nb))
print("Classification Report:\n", classification_report(y_test, y_pred_nb))

# Initialiser le classifieur SVM
svm = SVC(kernel='linear', random_state=42)  # Utilisez 'rbf' ou d'autres kernels si nécessaire

# Entraîner le modèle
svm.fit(X_train, y_train)

# Prédire les classes
y_pred_svm = svm.predict(X_test)

# Évaluation
print("SVM - Accuracy:", accuracy_score(y_test, y_pred_svm))
print("Classification Report:\n", classification_report(y_test, y_pred_svm))

# Initialiser le réseau de neurones
mlp = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)

# Entraîner le modèle
mlp.fit(X_train, y_train)

# Prédire les classes
y_pred_mlp = mlp.predict(X_test)

# Évaluation
print("Réseau de Neurones - Accuracy:", accuracy_score(y_test, y_pred_mlp))
print("Classification Report:\n", classification_report(y_test, y_pred_mlp))

conf_matrix = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=mlp.classes_)
disp.plot()
