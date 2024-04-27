#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import metrics



# In[ ]:


# CSV dosyasını okuma
df = pd.read_csv(r'C:\Users\kubra\Desktop\data\dataset.csv'


# In[ ]:


# Veri ön işleme adımları
# Burada, veri ön işleme adımlarını yapmanız gerekebilir: eksik veri kontrolü, kodlama işlemleri, özellik seçimi vb.

# Özellikleri ve hedef değişkeni belirleme
X = df.drop('crypto_name', axis=1)  # Özellikler
y = df['crypto_name']  # Hedef değişke


# In[ ]:


# Label Encoding
le = LabelEncoder()
X_encoded = X.apply(le.fit_transform)

# Verileri ölçeklendirme
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)


# In[ ]:


# Veriyi eğitim ve test setlerine bölmek
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)


# In[ ]:


# Logistic Regression
model_lr = LogisticRegression(class_weight="balanced", multi_class='multinomial', solver='lbfgs', max_iter=200)
model_lr.fit(X_train, y_train)
lr_predictions = model_lr.predict(X_test)
print("Logistic Regression Classification Report:\n", metrics.classification_report(y_test, lr_predictions))


# In[ ]:


# Decision Tree
model_dt = DecisionTreeClassifier(criterion='entropy', random_state=0, max_depth=200)
model_dt.fit(X_train, y_train)
dt_predictions = model_dt.predict(X_test)
print("Decision Tree Classification Report:\n", metrics.classification_report(y_test, dt_predictions))


# In[ ]:


# KNN
model_knn = KNeighborsClassifier(n_neighbors=200, n_jobs=-1)  # n_jobs=-1 for parallel processing
model_knn.fit(X_train, y_train)
knn_predictions = model_knn.predict(X_test)
print("KNN Classification Report:\n", metrics.classification_report(y_test, knn_predictions))


# In[ ]:


# Random Forest
model_rf = RandomForestClassifier(n_estimators=200, random_state=0, class_weight="balanced",  n_jobs=-1)
model_rf.fit(X_train, y_train)
rf_predictions = model_rf.predict(X_test)
print("Random Forest Classification Report:\n", metrics.classification_report(y_test, rf_predictions))


# In[ ]:


# MLP
model_mlp = MLPClassifier(hidden_layer_sizes=(6, 6, 6, 6), max_iter=200)
model_mlp.fit(X_train, y_train)
mlp_predictions = model_mlp.predict(X_test)
print("MLP Classification Report:\n", metrics.classification_report(y_test, mlp_predictions))

