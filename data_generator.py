
import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
import json

class FinancialDataGenerator:
    def __init__(self, db_path='financial_data.db'):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.setup_database()

    def setup_database(self):
        """Create database tables using the schema"""
        with open('database_setup.sql', 'r') as f:
            schema = f.read()

        # Execute schema creation
        cursor = self.conn.cursor()
        for statement in schema.split(';'):
            if statement.strip():
                cursor.execute(statement)
        self.conn.commit()
        print("Database tables created successfully")

    def generate_users(self, n_users=500):
        """Generate realistic user profiles"""
        users_data = []

        for i in range(n_users):
            # Generate realistic age distribution (22-65, skewed younger)
            age = int(np.random.beta(2, 3) * 43 + 22)

            # Generate income based on age and normal distribution
            base_income = 30000 + (age - 22) * 1200  # Income increases with age
            income = max(25000, np.random.normal(base_income, 15000))

            # Credit score correlates with income and age
            credit_score_base = min(850, 580 + (income / 1000) + (age - 22) * 2)
            credit_score = int(max(300, np.random.normal(credit_score_base, 80)))

            # Employment status based on age
            employment_options = ['full_time', 'part_time', 'self_employed', 'unemployed']
            employment_weights = [0.7, 0.15, 0.1, 0.05] if age > 25 else [0.5, 0.3, 0.1, 0.1]
            employment = np.random.choice(employment_options, p=employment_weights)

            user_data = (
                f'user_{i+1}',
                f'user{i+1}@email.com', 
                age,
                round(income, 2),
                employment,
                credit_score
            )
            users_data.append(user_data)

        # Insert into database
        cursor = self.conn.cursor()
        cursor.executemany(
            \'INSERT INTO users (username, email, age, income, employment_status, credit_score) VALUES (?, ?, ?, ?, ?, ?)\',
            users_data
        )
        self.conn.commit()
        print(f"Generated {n_users} user profiles")
        return users_data

    def generate_debt_accounts(self):
        """Generate debt accounts for users"""
        cursor = self.conn.cursor()
        users = cursor.execute('SELECT user_id, income, credit_score FROM users').fetchall()

        debt_types = {
            'credit_card': {'min_rate': 0.15, 'max_rate': 0.29, 'min_balance': 500, 'max_balance': 15000},
            'student_loan': {'min_rate': 0.04, 'max_rate': 0.08, 'min_balance': 5000, 'max_balance': 80000},
            'car_loan': {'min_rate': 0.03, 'max_rate': 0.12, 'min_balance': 5000, 'max_balance': 45000},
            'personal_loan': {'min_rate': 0.06, 'max_rate': 0.20, 'min_balance': 2000, 'max_balance': 25000}
        }

        creditors = {
            'credit_card': ['Chase', 'Capital One', 'Citi', 'Bank of America', 'Wells Fargo'],
            'student_loan': ['Federal Direct', 'Sallie Mae', 'Navient', 'Great Lakes'],
            'car_loan': ['Toyota Financial', 'Ford Credit', 'Honda Finance', 'Chase Auto'],
            'personal_loan': ['Marcus', 'SoFi', 'LightStream', 'Prosper']
        }

        accounts_data = []

        for user_id, income, credit_score in users:
            # Determine number of debts based on income and credit score
            debt_probability = min(0.9, (income / 100000) + (credit_score / 1000))
            n_debts = np.random.poisson(max(1, debt_probability * 3))

            for _ in range(min(n_debts, 5)):  # Max 5 debts per person
                debt_type = np.random.choice(list(debt_types.keys()), 
                                           p=[0.5, 0.2, 0.2, 0.1])

                debt_config = debt_types[debt_type]

                # Interest rate inversely correlated with credit score
                rate_factor = max(0.5, (850 - credit_score) / 550)
                interest_rate = debt_config['min_rate'] + (debt_config['max_rate'] - debt_config['min_rate']) * rate_factor

                # Balance correlates with income
                balance_factor = min(2.0, income / 50000)
                max_balance = min(debt_config['max_balance'], income * 0.4)  # Don't exceed 40% of income
                current_balance = np.random.uniform(debt_config['min_balance'], max_balance)

                # Original balance was higher
                original_balance = current_balance * np.random.uniform(1.1, 2.0)

                # Minimum payment calculation
                if debt_type == 'credit_card':
                    min_payment = max(25, current_balance * 0.02)  # 2% minimum
                    credit_limit = current_balance * np.random.uniform(1.2, 3.0)
                else:
                    # Other installment loans
                    term_months = {'student_loan': 120, 'car_loan': 60, 'personal_loan': 48}[debt_type]
                    monthly_rate = interest_rate / 12
                    min_payment = current_balance * (monthly_rate * (1 + monthly_rate)**term_months) / ((1 + monthly_rate)**term_months - 1)
                    credit_limit = None

                creditor = np.random.choice(creditors[debt_type])

                account_data = (
                    user_id,
                    debt_type,
                    creditor,
                    round(current_balance, 2),
                    round(original_balance, 2),
                    round(interest_rate, 4),
                    round(min_payment, 2),
                    round(credit_limit, 2) if credit_limit else None
                )
                accounts_data.append(account_data)

        # Insert into database
        cursor.executemany(
            \'INSERT INTO debt_accounts (user_id, debt_type, creditor_name, current_balance, original_balance, interest_rate, minimum_payment, credit_limit) VALUES (?, ?, ?, ?, ?, ?, ?, ?)\',
            accounts_data
        )
        self.conn.commit()
        print(f"Generated {len(accounts_data)} debt accounts")

    def close(self):
        self.conn.close()

# Usage example
if __name__ == "__main__":
    generator = FinancialDataGenerator()
    generator.generate_users(500)
    generator.generate_debt_accounts() 
    generator.close()
    print("Synthetic financial dataset created successfully!")
