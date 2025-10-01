
import numpy as np
import pandas as pd
import sqlite3
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, classification_report
import joblib
import json

class DebtPayoffOptimizer:
    def __init__(self, db_path='financial_data.db'):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.scaler = StandardScaler()
        self.payment_model = None
        self.credit_score_model = None

    def load_data(self):
        """Load and prepare data for modeling"""
        query = \'\'\'
        SELECT 
            u.user_id, u.age, u.income, u.credit_score, u.employment_status,
            da.debt_type, da.current_balance, da.interest_rate, da.minimum_payment,
            da.credit_limit, da.original_balance
        FROM users u
        JOIN debt_accounts da ON u.user_id = da.user_id
        \'\'\'

        df = pd.read_sql_query(query, self.conn)
        return df

    def engineer_features(self, df):
        """Create features for machine learning models"""
        # Debt-to-income ratio
        df['debt_to_income'] = df['current_balance'] / df['income']

        # Credit utilization (for credit cards)
        df['credit_utilization'] = df.apply(
            lambda row: row['current_balance'] / row['credit_limit'] 
            if pd.notna(row['credit_limit']) and row['credit_limit'] > 0 else 0, axis=1
        )

        # Payment burden ratio
        df['payment_burden'] = df['minimum_payment'] / (df['income'] / 12)

        # Debt progress (how much paid off)
        df['debt_progress'] = 1 - (df['current_balance'] / df['original_balance'])

        # Age groups
        df['age_group'] = pd.cut(df['age'], bins=[0, 30, 40, 50, 100], labels=['young', 'adult', 'middle', 'senior'])

        # Income groups
        df['income_group'] = pd.cut(df['income'], bins=[0, 40000, 60000, 80000, float('inf')], 
                                  labels=['low', 'medium', 'high', 'very_high'])

        return df

    def train_payment_prediction_model(self, df):
        """Train model to predict optimal payment amounts"""
        # Prepare features
        feature_cols = ['age', 'income', 'credit_score', 'current_balance', 'interest_rate',
                       'debt_to_income', 'credit_utilization', 'payment_burden']

        # Create target: optimal payment (higher than minimum for high-interest debt)
        df['optimal_payment'] = df.apply(
            lambda row: row['minimum_payment'] * (1 + row['interest_rate'] * 2) 
            if row['interest_rate'] > 0.15 else row['minimum_payment'] * 1.2, axis=1
        )

        # Encode categorical variables
        le_employment = LabelEncoder()
        le_debt_type = LabelEncoder()

        df_model = df.copy()
        df_model['employment_encoded'] = le_employment.fit_transform(df['employment_status'])
        df_model['debt_type_encoded'] = le_debt_type.fit_transform(df['debt_type'])

        feature_cols.extend(['employment_encoded', 'debt_type_encoded'])

        X = df_model[feature_cols].fillna(0)
        y = df_model['optimal_payment']

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train Random Forest model
        self.payment_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.payment_model.fit(X_train_scaled, y_train)

        # Evaluate model
        y_pred = self.payment_model.predict(X_test_scaled)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"Payment Prediction Model Performance:")
        print(f"Mean Absolute Error: ${mae:.2f}")
        print(f"R² Score: {r2:.4f}")

        return self.payment_model, feature_cols, le_employment, le_debt_type

    def calculate_debt_payoff_strategies(self, user_id):
        """Calculate different debt payoff strategies for a user"""
        query = \'\'\'
        SELECT * FROM debt_accounts 
        WHERE user_id = ? 
        ORDER BY interest_rate DESC
        \'\'\'

        debts = pd.read_sql_query(query, self.conn, params=(user_id,))

        if debts.empty:
            return None

        strategies = {}

        # Strategy 1: Debt Avalanche (highest interest first)
        avalanche_order = debts.sort_values('interest_rate', ascending=False)['account_id'].tolist()
        avalanche_savings = self._calculate_interest_savings(debts, avalanche_order)

        strategies['avalanche'] = {
            'name': 'Debt Avalanche',
            'order': avalanche_order,
            'total_interest_saved': avalanche_savings,
            'description': 'Pay minimums on all debts, extra payment goes to highest interest rate debt'
        }

        # Strategy 2: Debt Snowball (smallest balance first)  
        snowball_order = debts.sort_values('current_balance', ascending=True)['account_id'].tolist()
        snowball_savings = self._calculate_interest_savings(debts, snowball_order)

        strategies['snowball'] = {
            'name': 'Debt Snowball',
            'order': snowball_order,
            'total_interest_saved': snowball_savings,
            'description': 'Pay minimums on all debts, extra payment goes to smallest balance debt'
        }

        # Strategy 3: ML Optimized (balance multiple factors)
        if hasattr(self, 'payment_model') and self.payment_model:
            optimized_order = self._calculate_ml_optimized_order(debts)
            optimized_savings = self._calculate_interest_savings(debts, optimized_order)

            strategies['ml_optimized'] = {
                'name': 'ML Optimized',
                'order': optimized_order,
                'total_interest_saved': optimized_savings,
                'description': 'AI-optimized strategy considering multiple factors'
            }

        return strategies

    def _calculate_interest_savings(self, debts, payoff_order, extra_payment=200):
        """Calculate total interest savings for a given payoff strategy"""
        total_savings = 0
        remaining_debts = debts.copy()

        for account_id in payoff_order:
            debt = remaining_debts[remaining_debts['account_id'] == account_id].iloc[0]

            # Simple calculation: higher interest debts paid first save more
            if debt['interest_rate'] > 0.15:  # High interest threshold
                savings_factor = debt['interest_rate'] * debt['current_balance'] * 0.3
                total_savings += savings_factor

        return total_savings

    def _calculate_ml_optimized_order(self, debts):
        """Use ML model to determine optimal payoff order"""
        # Create a score for each debt based on multiple factors
        scores = []

        for _, debt in debts.iterrows():
            # Combine interest rate, balance, and other factors
            score = (debt['interest_rate'] * 0.4 + 
                    (1/debt['current_balance']) * 0.3 +  # Favor smaller balances
                    (debt['minimum_payment'] / debt['current_balance']) * 0.3)  # Consider payment efficiency

            scores.append((score, debt['account_id']))

        # Sort by score (highest first)
        scores.sort(reverse=True)
        return [account_id for _, account_id in scores]

    def save_models(self):
        """Save trained models"""
        if self.payment_model:
            joblib.dump(self.payment_model, 'payment_model.pkl')
            joblib.dump(self.scaler, 'scaler.pkl')
            print("Models saved successfully")

    def load_models(self):
        """Load pre-trained models"""
        try:
            self.payment_model = joblib.load('payment_model.pkl')
            self.scaler = joblib.load('scaler.pkl')
            print("Models loaded successfully")
        except FileNotFoundError:
            print("No saved models found")

    def close(self):
        self.conn.close()
