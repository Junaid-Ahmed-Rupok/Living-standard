import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, learning_curve, cross_val_score
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


df = joblib.load('df.pkl') # loading the cleaned dataframe
encoder = joblib.load('encoder.pkl') # loading the encoder

x = df.drop(columns=['Human Development Index']) # features of the model
y = df['Human Development Index'] # label of the model

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42) # splitting the dataframe

# feature selection

# adaboost 
ad_demo_model = AdaBoostRegressor(n_estimators=200, learning_rate=0.2, random_state=42)
ad_demo_model.fit(x_train, y_train)
ad_feature_importance = pd.Series(ad_demo_model.feature_importances_, index=x_train.columns).sort_values(ascending=False).head(10)
ad_features = ad_feature_importance.index.tolist() # ad features

# xgboost
xg_demo_model = XGBRegressor(n_estimators=200, learning_rate=0.2, random_state=42)
xg_demo_model.fit(x_train, y_train)
xg_feature_importance = pd.Series(xg_demo_model.feature_importances_, index=x_train.columns).sort_values(ascending=False).head(10)
xg_features = xg_feature_importance.index.tolist() # xg features

# random forest
rf_demo_model = RandomForestRegressor(random_state=42)
rf_demo_model.fit(x_train, y_train)
rf_feature_importance = pd.Series(rf_demo_model.feature_importances_, index=x_train.columns).sort_values(ascending=False).head(10)
rf_features = rf_feature_importance.index.tolist() # rf features

# decision tree
dt_demo_model = DecisionTreeRegressor(random_state=42)
dt_demo_model.fit(x_train, y_train)
dt_feature_importance = pd.Series(dt_demo_model.feature_importances_, index=x_train.columns).sort_values(ascending=False).head(10)
dt_features = dt_feature_importance.index.tolist() # dt features

# Lasso regression
lasso_model = Lasso(alpha=1.0)
lasso_model.fit(x_train, y_train)
lasso_coef = pd.Series(lasso_model.coef_, index=x_train.columns)
lasso_features = lasso_coef[lasso_coef != 0].index.tolist()

# defining the models
models = {
    'Adaboost': (AdaBoostRegressor(n_estimators=200, learning_rate=0.2, random_state=42), ad_features), 
    'XGBoost': (XGBRegressor(n_estimators=200, learning_rate=0.2, random_state=42), xg_features), 
    'Decision Tree': (DecisionTreeRegressor(random_state=42), dt_features),
    'Random Forest': (RandomForestRegressor(random_state=42), rf_features), 
    'Lasso': (Lasso(alpha=1.0), lasso_features)
}


# regression analysis report
def regression_report(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mean_squared_error(y_true, y_pred),
        "R²": r2_score(y_true, y_pred)
    }

# finding the best model using cross-validation model
best_score, best_name, best_model, best_features = 0, None, None, None

for name, (model, features) in models.items():
    score = cross_val_score(model, x_train[features], y_train, cv=5).mean() # performing the cross-validation process
    
    if score > best_score:
        best_score, best_name, best_model, best_features = score, name, model, features # updating the values
    
    print(f"cross validation score of {name}: {score}")

print(f"The best model is {best_name.upper()} with {best_score} cross-validation score.")


# training handling
best_model.fit(x_train[best_features], y_train) # training the best model
predictions = best_model.predict(x_test[best_features]) # prediction using the best model
print('\n' * 3)
print(regression_report(y_test, predictions)) # report of the best model
print('\n' * 3)
joblib.dump(best_model, 'model.pkl') # dumping the best model


# overfitting evaluation 

def overfitting(model, features):
    train_predictions = model.predict(x_train[features])
    test_predictions = model.predict(x_test[features])
    
    train_accuracy = (y_train == train_predictions).mean()
    test_accuracy = (y_test == test_predictions).mean()
    
    if abs(train_accuracy - test_accuracy) > 0.1:
        print(f"------>Overfitting Warning.")
    else: 
        print(f"------>No significant Overfitting.")

# overfitting checking of the best model
overfitting(best_model, best_features)

print("/" * 3)
# overfitting checking for all other model
for name, (model, features) in models.items():
    model.fit(x_train[features], y_train)
    if name != best_name:
        print(f"{name} Overfitting checking:")
        overfitting(model, features) 
        print("/" * 2)

print("\n" * 3)

# learning curves
row, column = 0, 0
row_max, column_max = 2, 3
fig, ax = plt.subplots(row_max, column_max, figsize=(10, 10))

for name, (model, features) in models.items():
    train_sizes, training_score, testing_score = learning_curve(
            model, x_train[features], y_train, cv=5, scoring='r2', train_sizes=np.linspace(0.1, 1, 10)
        )
   
    if column == column_max:
        row += 1
        column = 0
    
    ax[row, column].set_xlabel('Train Sizes', color='blue')
    ax[row, column].set_ylabel('Accuracy Score', color='red')
    ax[row, column].set_title('Learning Curve', color='green')
    ax[row, column].plot(train_sizes, training_score.mean(1), color='green', label='Training Curve')
    ax[row, column].plot(train_sizes, testing_score.mean(1), color='red', label='testing score')
    ax[row, column].grid(True)
    ax[row, column].legend()
    
    column += 1

plt.tight_layout()
plt.show()
        

# feature importance graph
fig, bx = plt.subplots(row_max, column_max, figsize=(10, 10))
feature_importance_dictionary = {
    'Adaboost': ad_feature_importance, 'XGBoost': xg_feature_importance, 'Decision Tree': dt_feature_importance,
    'Random Forest': rf_feature_importance
}
row, column = 0, 0
for name, value in models.items():
    if column == column_max:
        row += 1
        column = 0
    
    if name in feature_importance_dictionary.keys():
        importance = feature_importance_dictionary[name]
        bx[row, column].set_xlabel('Feature', color='green')
        bx[row, column].set_ylabel('Feature Importance', color='green')
        bx[row, column].set_title('Feature Importance', color='black')
        importance.plot(kind='bar', ax=bx[row, column], color='blue')
        bx[row, column].tick_params(axis='x', rotation=45)
        bx[row, column].grid(True)
    
        column += 1

plt.tight_layout()
plt.show()


