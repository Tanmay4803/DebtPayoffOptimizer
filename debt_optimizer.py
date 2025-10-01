
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from models import DebtPayoffOptimizer
from data_generator import FinancialDataGenerator
import warnings
warnings.filterwarnings('ignore')

class SmartDebtPayoffOptimizer:
    def __init__(self, db_path='financial_data.db'):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.optimizer = DebtPayoffOptimizer(db_path)

    def setup_demo_data(self):
        """Create demo dataset with synthetic data"""
        print("Setting up demonstration dataset...")
        generator = FinancialDataGenerator(self.db_path)
        generator.generate_users(500)
        generator.generate_debt_accounts()
        generator.close()
        print("Demo data created successfully!")

    def analyze_user_debt_profile(self, user_id):
        """Analyze a user's complete debt profile"""
        print(f"\n{'='*50}")
        print(f"DEBT ANALYSIS FOR USER {user_id}")
        print(f"{'='*50}")

        # Get user information
        user_query = "SELECT * FROM users WHERE user_id = ?"
        user_info = pd.read_sql_query(user_query, self.conn, params=(user_id,))

        if user_info.empty:
            print("User not found!")
            return None

        user = user_info.iloc[0]
        print(f"User: {user['username']}")
        print(f"Age: {user['age']}, Income: ${user['income']:,.2f}")
        print(f"Credit Score: {user['credit_score']}")
        print(f"Employment: {user['employment_status']}")

        # Get debt accounts
        debts_query = "SELECT * FROM debt_accounts WHERE user_id = ?"
        debts = pd.read_sql_query(debts_query, self.conn, params=(user_id,))

        if debts.empty:
            print("No debt accounts found for this user.")
            return None

        print(f"\nDEBT PORTFOLIO ({len(debts)} accounts):")
        print("-" * 80)
        total_debt = 0
        total_min_payment = 0

        for _, debt in debts.iterrows():
            total_debt += debt['current_balance']
            total_min_payment += debt['minimum_payment']

            print(f"{debt['debt_type'].title()}: {debt['creditor_name']}")
            print(f"  Balance: ${debt['current_balance']:,.2f} at {debt['interest_rate']*100:.2f}% APR")
            print(f"  Minimum Payment: ${debt['minimum_payment']:,.2f}")
            if debt['credit_limit']:
                utilization = (debt['current_balance'] / debt['credit_limit']) * 100
                print(f"  Utilization: {utilization:.1f}%")
            print()

        print(f"TOTALS:")
        print(f"Total Debt: ${total_debt:,.2f}")
        print(f"Total Minimum Payments: ${total_min_payment:,.2f}")
        print(f"Debt-to-Income Ratio: {(total_debt/user['income']*100):.1f}%")

        return {
            'user_info': user,
            'debts': debts,
            'total_debt': total_debt,
            'total_min_payment': total_min_payment
        }

    def generate_payoff_recommendations(self, user_id, extra_monthly_payment=0):
        """Generate AI-powered debt payoff recommendations"""
        print(f"\n{'='*50}")
        print(f"PAYOFF STRATEGY RECOMMENDATIONS")
        print(f"{'='*50}")

        strategies = self.optimizer.calculate_debt_payoff_strategies(user_id)

        if not strategies:
            print("No debt accounts found for strategy calculation.")
            return None

        print(f"Extra monthly payment available: ${extra_monthly_payment:,.2f}")
        print()

        # Display each strategy
        for strategy_key, strategy in strategies.items():
            print(f"{strategy['name'].upper()}:")
            print(f"Description: {strategy['description']}")
            print(f"Estimated Interest Savings: ${strategy['total_interest_saved']:,.2f}")

            # Get debt details for ordered list
            debt_query = """
            SELECT account_id, debt_type, creditor_name, current_balance, interest_rate, minimum_payment
            FROM debt_accounts 
            WHERE account_id IN ({})
            """.format(','.join(['?' for _ in strategy['order']]))

            ordered_debts = pd.read_sql_query(debt_query, self.conn, params=strategy['order'])
            ordered_debts = ordered_debts.set_index('account_id').reindex(strategy['order']).reset_index()

            print("Payment Priority Order:")
            for i, (_, debt) in enumerate(ordered_debts.iterrows(), 1):
                print(f"  {i}. {debt['debt_type'].title()} - {debt['creditor_name']}")
                print(f"     ${debt['current_balance']:,.2f} at {debt['interest_rate']*100:.2f}% APR")
            print()

        # Save recommendation to database
        self._save_recommendation(user_id, strategies, extra_monthly_payment)

        return strategies

    def _save_recommendation(self, user_id, strategies, extra_payment):
        """Save the recommendation to the database"""
        # Get the best strategy (highest savings)
        best_strategy = max(strategies.values(), key=lambda x: x['total_interest_saved'])

        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO payoff_recommendations 
            (user_id, strategy_name, total_interest_saved, recommended_extra_payment, priority_order)
            VALUES (?, ?, ?, ?, ?)
        """, (
            user_id,
            best_strategy['name'],
            best_strategy['total_interest_saved'],
            extra_payment,
            str(best_strategy['order'])  # Convert list to string
        ))
        self.conn.commit()

    def train_ml_models(self):
        """Train machine learning models on the dataset"""
        print("\nTraining machine learning models...")

        # Load data
        df = self.optimizer.load_data()
        if df.empty:
            print("No data available for training. Please generate data first.")
            return

        print(f"Training on {len(df)} debt accounts from {df['user_id'].nunique()} users")

        # Engineer features
        df = self.optimizer.engineer_features(df)

        # Train payment prediction model
        self.optimizer.train_payment_prediction_model(df)

        # Save models
        self.optimizer.save_models()

        print("Model training completed successfully!")

    def batch_analysis(self, n_users=10):
        """Perform analysis on multiple users for demonstration"""
        print(f"\n{'='*60}")
        print(f"BATCH ANALYSIS OF {n_users} USERS")
        print(f"{'='*60}")

        # Get random users with debt
        query = """
        SELECT DISTINCT u.user_id 
        FROM users u 
        JOIN debt_accounts da ON u.user_id = da.user_id 
        ORDER BY RANDOM() 
        LIMIT ?
        """

        user_ids = pd.read_sql_query(query, self.conn, params=(n_users,))['user_id'].tolist()

        summary_data = []

        for user_id in user_ids:
            profile = self.analyze_user_debt_profile(user_id)
            if profile:
                strategies = self.generate_payoff_recommendations(user_id, extra_monthly_payment=200)

                if strategies:
                    best_strategy = max(strategies.values(), key=lambda x: x['total_interest_saved'])

                    summary_data.append({
                        'user_id': user_id,
                        'total_debt': profile['total_debt'],
                        'total_min_payment': profile['total_min_payment'],
                        'debt_to_income': (profile['total_debt'] / profile['user_info']['income']) * 100,
                        'credit_score': profile['user_info']['credit_score'],
                        'best_strategy': best_strategy['name'],
                        'potential_savings': best_strategy['total_interest_saved']
                    })

        # Create summary report
        if summary_data:
            summary_df = pd.DataFrame(summary_data)

            print(f"\n{'='*60}")
            print("SUMMARY STATISTICS")
            print(f"{'='*60}")
            print(f"Average Total Debt: ${summary_df['total_debt'].mean():,.2f}")
            print(f"Average Debt-to-Income Ratio: {summary_df['debt_to_income'].mean():.1f}%")
            print(f"Average Credit Score: {summary_df['credit_score'].mean():.0f}")
            print(f"Average Potential Savings: ${summary_df['potential_savings'].mean():,.2f}")

            print(f"\nStrategy Distribution:")
            strategy_counts = summary_df['best_strategy'].value_counts()
            for strategy, count in strategy_counts.items():
                print(f"  {strategy}: {count} users ({count/len(summary_df)*100:.1f}%)")

            return summary_df

        return None

    def close(self):
        """Close database connections"""
        self.conn.close()
        self.optimizer.close()

def main():
    """Main demonstration function"""
    app = SmartDebtPayoffOptimizer()

    print("Smart Debt Payoff Optimizer - AI-Powered Financial Tool")
    print("=" * 60)

    # Setup demo data
    app.setup_demo_data()

    # Train ML models
    app.train_ml_models()

    # Run batch analysis
    app.batch_analysis(5)

    print("\n" + "="*60)
    print("DEMONSTRATION COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\nProject Features Demonstrated:")
    print("✓ AI-powered debt payoff strategy optimization")
    print("✓ Multiple strategy comparison (Avalanche, Snowball, ML-Optimized)")
    print("✓ Comprehensive user financial profiling")
    print("✓ Statistical analysis and reporting")
    print("✓ Machine learning model training and prediction")

    app.close()

if __name__ == "__main__":
    main()
