from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
data = pd.read_csv('/Users/dany/Desktop/Новая папка/stroke/healthcare-dataset-stroke-data.csv')

#set BMI columns

data['bmi'] = data['bmi'].fillna(data['bmi'].median())
num_f = ['age', 'avg_glucose_level', 'bmi']
cat_f = ['hypertension', 'gender', 'heart_disease',	'ever_married',	'work_type', 'Residence_type', 'smoking_status']

num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median'))
])

cat_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers=[('num', num_transformer, num_f), ('cat', cat_transformer, cat_f)])

X = data.drop(columns=['stroke'])
Y = data['stroke']

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size= 0.2, random_state=42)
X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)
print(X_train)

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, roc_auc_score
models = {'log_reg': LogisticRegression(max_iter=100), 'rand_forest': RandomForestClassifier(n_estimators=50, random_state=42), 'SVM': SVC(probability=True, random_state=42)}

for  name, model in models.items():
    model.fit(X_train, Y_train)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:,1]
    # print(f'\n{name}')
    # print(classification_report(Y_test, y_pred))
    # print(roc_auc_score(Y_test, y_pred_proba))

rf_model = models['rand_forest']
feature_imp = rf_model.feature_importances_
feature_names = num_f + list(preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out())
df = pd.DataFrame({'feature': feature_names, 'importance' : feature_imp})
df = df.sort_values(by='importance', ascending=False)
print(df)

plt.figure(figsize=(12,6))
sns.barplot(x='importance', y='feature', data=df.head(10))
plt.title('10 most important features to have stroke')
plt.show()