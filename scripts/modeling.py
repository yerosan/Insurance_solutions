import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import RFE
from xgboost import XGBRegressor
import numpy as np
import shap

class Model():
    def __init__(self, df):
        self.df=df
        
    def droping_inrrelevant(self):
        df=self.df
        # Drop unnecessary columns
        irrelevantColumns=['UnderwrittenCoverID', 'PolicyID',"Language","Country","MainCrestaZone","SubCrestaZone","ItemType", "PostalCode",
                           "bodytype","ExcessSelected","CoverCategory","CoverType","StatutoryClass","StatutoryRiskType","mmcode"]
        df.drop(columns=irrelevantColumns, inplace=True)
        return df
    

    def useful_informationExtraction(self,data):
        # Extracting useful information from TransactionMonth
        data['TransactionMonth'] = pd.to_datetime(data['TransactionMonth'])
        data['Transaction_Year'] = data['TransactionMonth'].dt.year
        data['Transaction_Month'] = data['TransactionMonth'].dt.month

        # Vehicle Age (VehicleIntroDate)
        data['VehicleIntroDate'] = pd.to_datetime(data['VehicleIntroDate'], format='%Y-%m-%d', errors='coerce')
        data['VehicleAge'] = data['Transaction_Year'] - data['VehicleIntroDate'].dt.year


        return data


    def categoricalEndcoding(self,data):

        # Replace comma with period and then convert to float
        data['CapitalOutstanding'] = data['CapitalOutstanding'].str.replace(',', '.').astype(float)

        # Categorical encoding using one-hot encoding
        cat_data=data.select_dtypes("object")
        cat_columns=cat_data.columns
         # List of columns to exclude
        exclude_columns = ['make', 'Model']

        # Filter categorical_columns by excluding the columns in exclude_columns
        filtered_columns = [col for col in cat_columns if col not in exclude_columns]

        df = pd.get_dummies(data, columns=filtered_columns, drop_first=True)
        
       

        # LabelEncoding can also be used for make and model columns
        
        le = LabelEncoder()
        df['make'] = le.fit_transform(df['make'])
        df['Model'] = le.fit_transform(df['Model'])

        # Convert all boolean columns to integers
        df[df.select_dtypes(include=['bool']).columns] = df.select_dtypes(include=['bool']).astype(int)


        return df

    

    def feature_scaling(self, data):
        # Features to scale
        scaler = StandardScaler()
        numerical_columns = ['TotalPremium',"CalculatedPremiumPerTerm", "CapitalOutstanding",'TotalClaims', 'SumInsured', 'cubiccapacity', 'kilowatts']

        # Apply scaling
        data[numerical_columns] = scaler.fit_transform(data[numerical_columns])
        data=data.drop(columns="TransactionMonth")
        return data
    

    def Data_spliting(self, data):
        # Separate the target variable from features
        data=data.head(200000)
        X = data.drop(columns=['TotalClaims']).reset_index(drop=True)  # Features
        y = data['TotalClaims']  # Target variable
        # X = df.drop(columns='target_column')) 
        # Split data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        return X_train, X_test, y_train, y_test 

    def feature_selection(self,X_train,y_train, X_test):
        

        # Initialize the model
        model = RandomForestRegressor()

        # Perform RFE to select the top 10 features
        rfe = RFE(estimator=model, n_features_to_select=10)
        rfe.fit(X_train, y_train)

        # Get the selected features
        selected_features = X_train.columns[rfe.support_]
        print("Selected Features: ", selected_features)

        # Proceed to model training using the selected features
        X_train_selected = X_train[selected_features]
        X_test_selected = X_test[selected_features]

        return selected_features, X_train_selected, X_test_selected
    

    def linearRegression(self,X_train_selected,y_train,X_test_selected,y_test):
        # Initialize and train Linear Regression model
        lr_model = LinearRegression()
        lr_model.fit(X_train_selected, y_train)

        # Predictions on the test set
        y_pred_lr = lr_model.predict(X_test_selected)

        # Evaluation
        rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
        mae_lr = mean_absolute_error(y_test, y_pred_lr)

        print(f"Linear Regression RMSE: {rmse_lr}")
        print(f"Linear Regression MAE: {mae_lr}")
        return lr_model,rmse_lr, mae_lr



    def random_forest(self,X_train_selected,y_train,X_test_selected,y_test):

        # Initialize and train Random Forest model
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train_selected, y_train)

        # Predictions on the test set
        y_pred_rf = rf_model.predict(X_test_selected)

        # Evaluation
        rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
        mae_rf = mean_absolute_error(y_test, y_pred_rf)

        print(f"Random Forest RMSE: {rmse_rf}")
        print(f"Random Forest MAE: {mae_rf}")
        return rf_model,rmse_rf, mae_rf


    def xgboost(self,X_train_selected,y_train,X_test_selected,y_test ):
        # Initialize and train XGBoost model
        xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
        xgb_model.fit(X_train_selected, y_train)

        # Predictions on the test set
        y_pred_xgb = xgb_model.predict(X_test_selected)

        # Evaluation
        rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
        mae_xgb = mean_absolute_error(y_test, y_pred_xgb)

        print(f"XGBoost RMSE: {rmse_xgb}")
        print(f"XGBoost MAE: {mae_xgb}")

        return xgb_model,rmse_xgb, mae_xgb
    

    def model_comparesion(self,rmse_lr,rmse_rf,rmse_xgb,mae_lr,mae_rf,mae_xgb):
        # Comparison of RMSE and MAE across models
        performance = {
            'Model': ['Linear Regression', 'Random Forest', 'XGBoost'],
            'RMSE': [rmse_lr, rmse_rf, rmse_xgb],
            'MAE': [mae_lr, mae_rf, mae_xgb]
        }

        performance_df = pd.DataFrame(performance)
        print(performance_df)


    def besf_randomForest(self,rf_model,X_train_selected,y_train,X_test_selected, y_test):

        # Define parameter grid
        param_grid_rf = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30],
            'min_samples_split': [2, 5, 10]
        }

        # Initialize GridSearchCV
        grid_rf = GridSearchCV(estimator=rf_model, param_grid=param_grid_rf, scoring='neg_mean_squared_error', cv=5, verbose=2)

        # Fit the model
        grid_rf.fit(X_train_selected, y_train)

        # Best parameters
        print(f"Best parameters for Random Forest: {grid_rf.best_params_}")

        # Train with best parameters and evaluate
        best_rf_model = grid_rf.best_estimator_
        y_pred_best_rf = best_rf_model.predict(X_test_selected)

        rmse_best_rf = np.sqrt(mean_squared_error(y_test, y_pred_best_rf))
        mae_best_rf = mean_absolute_error(y_test, y_pred_best_rf)

        print(f"Tuned Random Forest RMSE: {rmse_best_rf}")
        print(f"Tuned Random Forest MAE: {mae_best_rf}")

        return best_rf_model


    def best_xgboost(self,xgb_model,X_train_selected,y_train,X_test_selected,y_test):
        param_grid_xgb = {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7]
        }

        # Initialize GridSearchCV
        grid_xgb = GridSearchCV(estimator=xgb_model, param_grid=param_grid_xgb, scoring='neg_mean_squared_error', cv=5, verbose=2)

        # Fit the model
        grid_xgb.fit(X_train_selected, y_train)

        # Best parameters
        print(f"Best parameters for XGBoost: {grid_xgb.best_params_}")

        # Train with best parameters and evaluate
        best_xgb_model = grid_xgb.best_estimator_
        y_pred_best_xgb = best_xgb_model.predict(X_test_selected)

        rmse_best_xgb = np.sqrt(mean_squared_error(y_test, y_pred_best_xgb))
        mae_best_xgb = mean_absolute_error(y_test, y_pred_best_xgb)

        print(f"Tuned XGBoost RMSE: {rmse_best_xgb}")
        print(f"Tuned XGBoost MAE: {mae_best_xgb}")
        return best_xgb_model


  
    def shap_explanier(self,best_xgb_model,X_test_selected):
        # Initialize SHAP explainer for XGBoost
        explainer = shap.Explainer(best_xgb_model)
        shap_values = explainer(X_test_selected)

        # Plot summary of SHAP values
        shap.summary_plot(shap_values, X_test_selected)


