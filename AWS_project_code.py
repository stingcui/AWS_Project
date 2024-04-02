import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from scipy.stats import uniform, randint
from sklearn.model_selection import train_test_split
#%%
#Importing the dataset
train_file_path = "/Users/stingcui/PycharmProjects/AWS/train.csv"
houseprice_df = pd.read_csv(train_file_path)

test_file_path = "/Users/stingcui/PycharmProjects/AWS/test.csv"
houseprice_test_df = pd.read_csv(test_file_path)
#%%
#Drop the Id column
houseprice_df = houseprice_df.drop('Id', axis=1)
# return number of rows and columns
print(houseprice_df.shape)
#%%
#X and y variables
X = houseprice_df.drop('SalePrice', axis=1)
y = houseprice_df['SalePrice']
print(X.head())
# %%
#Train and test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Select numerical features
numerical_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Select categorical features
categorical_features = X_train.select_dtypes(include=['object']).columns.tolist()

# Numerical pipeline
numerical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Categorical pipeline
categorical_pipeline = Pipeline(steps = [
    ('imputer', SimpleImputer(strategy = 'most_frequent', fill_value = 'missing')),
    ('onehot', OneHotEncoder(handle_unknown = 'ignore'))  # Set handle_unknown to 'ignore'
])

# perform preprocessing to the data
preprocessor = ColumnTransformer([
    ('numerical', numerical_pipeline, numerical_features),
    ('categorical', categorical_pipeline, categorical_features)
])
#%%
# deploy the preprocessing pipeline
X_train = preprocessor.fit_transform(X_train) # fit_transform on the training set
X_val = preprocessor.transform(X_val) # fit_transform on the validation set
X_test = preprocessor.transform(X_test) # fit_transform on the test set
# %%
train = pd.concat([X_train, y_train], axis=1)
test = pd.concat([X_test, y_test], axis=1)
#%%
import pandas as pd

# Convert sparse matrix to DataFrame or Series
X_train_df = pd.DataFrame(X_train.toarray(), columns=preprocessor.get_feature_names_out())
X_val_df = pd.DataFrame(X_val.toarray(), columns=preprocessor.get_feature_names_out())
X_test_df = pd.DataFrame(X_test.toarray(), columns=preprocessor.get_feature_names_out())

# Concatenate with y_train and y_test
train = pd.concat([X_train_df, y_train.reset_index(drop=True)], axis=1)
val = pd.concat([X_val_df, y_val.reset_index(drop=True)], axis=1)
test = pd.concat([X_test_df, y_test.reset_index(drop=True)], axis=1)

#%%
# XGBoost pipeline
xgb_pipeline = Pipeline([
    ('preprocessing', ColumnTransformer([
        ('numerical', numerical_pipeline, numerical_features),
        ('categorical', categorical_pipeline, categorical_features),
    ])),
    ('model', xgb.XGBRegressor())
])

## FIRST RANDOMIZED SEARCH -------------------------------------------------------------------

# Parameter grid for RandomizedSearchCV
param_grid = {
    'model__n_estimators': [100, 200, 300, 400, 500],
    'model__max_depth': [3, 4, 5, 6, 7],
    'model__learning_rate': [0.01, 0.1, 0.2, 0.3],
    'model__subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
    'model__colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
    'model__gamma': [0, 1, 5],
    'model__reg_alpha': [0, 0.1, 0.5, 1.0],
    'model__reg_lambda': [0, 0.1, 0.5, 1.0]
}

# RandomizedSearchCV
random_search = RandomizedSearchCV(xgb_pipeline, param_distributions=param_grid, scoring='r2', n_iter=100, cv=5, verbose = 1, random_state=42)
random_search.fit(X_train, y_train)

# Extract the best model
best_model = random_search.best_estimator_

print("Best Score:", random_search.best_score_)
print("Best Model:", best_model)

## SECOND RANDOMIZED SEARCH -------------------------------------------------------------------

# Define a more focused parameter grid based on the best model
param_grid_focused = {
    'model__learning_rate': [0.1, 0.2, 0.3, 0.4],
    'model__max_depth': [4, 5, 6, 7],
    'model__n_estimators': [150, 200, 250, 300],
}

# Create the GridSearchCV object
grid_search = GridSearchCV(best_model, param_grid=param_grid_focused, cv=5, verbose=1)

# Fit the GridSearchCV object to the data
grid_search.fit(X_train, y_train)

# Print the best model and best score
print("Best model:", grid_search.best_estimator_)
print("Best score:", grid_search.best_score_)

## THIRD RANDOMIZED SEARCH -------------------------------------------------------------------

# Define a parameter distribution for the random search
param_dist = {
    'model__learning_rate': uniform(0.1, 0.3),
    'model__max_depth': randint(3, 7),
    'model__n_estimators': randint(150, 250),
}

# Create the RandomizedSearchCV object
random_search = RandomizedSearchCV(best_model, param_distributions=param_dist, scoring='r2', n_iter=100, cv=5, verbose=1, random_state=42)

# Fit the RandomizedSearchCV object to the data
random_search.fit(X_train, y_train)

# Save the best model and best score
best_model_xgb = random_search.best_estimator_
best_score_xgb = random_search.best_score_

# Print the best model and best score
print("Best model:", best_model_xgb)
print("Best score:", best_score_xgb)

# Compute the generalization score
generalization_score = best_model_xgb.score(X_test, y_test)
print("Generalization score:", generalization_score)
