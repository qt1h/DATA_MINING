import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.discriminant_analysis import (LinearDiscriminantAnalysis,
                                           QuadraticDiscriminantAnalysis)
from sklearn.ensemble import (AdaBoostClassifier, BaggingClassifier,
                              ExtraTreesClassifier, GradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.linear_model import (LogisticRegression,
                                  PassiveAggressiveClassifier, Perceptron,
                                  RidgeClassifier)
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, confusion_matrix)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# Charger le dataset
file_path = 'dataset/star_classification.csv'
data = pd.read_csv(file_path)

# Créer un dossier pour enregistrer les matrices de confusion
output_dir = "confusion_matrices"
os.makedirs(output_dir, exist_ok=True)

# Exploration des données
print("Aperçu des données:")
print(data.head())
print("\nInformations sur le dataset:")
print(data.info())

# Vérification des valeurs manquantes
print("\nValeurs manquantes par colonne:")
print(data.isnull().sum())

# Préparation des données
X = data.drop('class', axis=1)
y = data['class']

# Normalisation des données
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Division en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Liste des modèles à évaluer
models = [
    ("K-Nearest Neighbors", KNeighborsClassifier(n_neighbors=5)),
    ("Decision Tree", DecisionTreeClassifier(random_state=42)),
    ("Naïve Bayes", GaussianNB()),
    ("Support Vector Machine", SVC(kernel='linear', random_state=42)),
    ("Multi-Layer Perceptron", MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)),
    ("Random Forest", RandomForestClassifier(n_estimators=100, random_state=42)),
    ("Gradient Boosting", GradientBoostingClassifier(random_state=42)),
    ("Logistic Regression", LogisticRegression(max_iter=500, random_state=42)),
    ("AdaBoost", AdaBoostClassifier(random_state=42)),
    ("Extra Trees", ExtraTreesClassifier(n_estimators=100, random_state=42)),
    ("Bagging Classifier", BaggingClassifier(estimator=DecisionTreeClassifier(), n_estimators=100, random_state=42)),
    ("Linear Discriminant Analysis", LinearDiscriminantAnalysis()),
    ("Quadratic Discriminant Analysis", QuadraticDiscriminantAnalysis()),
    ("Ridge Classifier", RidgeClassifier(random_state=42)),
    ("Perceptron", Perceptron(max_iter=1000, random_state=42)),
    ("Passive Aggressive Classifier", PassiveAggressiveClassifier(max_iter=1000, random_state=42))
]

# Fonction utilitaire pour évaluer les modèles
results = {}
def evaluate_model(model, model_name):
    try:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\n{model_name} - Accuracy: {accuracy:.4f}")
        print("Classification Report:\n", classification_report(y_test, y_pred))
        results[model_name] = accuracy

        # Matrice de confusion
        conf_matrix = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=np.unique(y))
        disp.plot()
        plt.title(f"Matrice de confusion - {model_name}")
        output_path = os.path.join(output_dir, f"confusion_matrix_{model_name.replace(' ', '_')}.png")
        plt.savefig(output_path)
        print(f"Matrice de confusion enregistrée dans : {output_path}")
        plt.close()
        
    except ValueError as e:
        print(f"Erreur avec le modèle {model_name}: {e}")

# Évaluer tous les modèles
for model_name, model in models:
    evaluate_model(model, model_name)

# Comparaison des modèles
def plot_model_comparison(results):
    results_df = pd.DataFrame(list(results.items()), columns=['Algorithm', 'Accuracy'])
    results_df = results_df.sort_values(by='Accuracy', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=results_df, x='Accuracy', y='Algorithm', palette='viridis')
    plt.title("Comparaison des modèles - Précision", fontsize=16)
    plt.xlabel("Précision", fontsize=12)
    plt.ylabel("Algorithme", fontsize=12)
    plt.xlim(0, 1)
    plt.tight_layout()
    plt.show()

plot_model_comparison(results)

# Pairplot des caractéristiques sélectionnées (prend du temps)
# selected_columns = ['r', 'i', 'alpha', 'delta', 'redshift', 'class']
# sns.pairplot(data[selected_columns], hue='class', palette='Set1')
# plt.show()
