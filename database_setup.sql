
-- Smart Debt Payoff Optimizer Database Schema

-- Users table to store customer information
CREATE TABLE users (
    user_id INTEGER PRIMARY KEY AUTOINCREMENT,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    age INTEGER,
    income DECIMAL(10, 2),
    employment_status VARCHAR(30),
    credit_score INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Debt accounts table
CREATE TABLE debt_accounts (
    account_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    debt_type VARCHAR(30) NOT NULL, -- credit_card, student_loan, car_loan, etc.
    creditor_name VARCHAR(100),
    current_balance DECIMAL(10, 2) NOT NULL,
    original_balance DECIMAL(10, 2),
    interest_rate DECIMAL(5, 4) NOT NULL, -- APR as decimal
    minimum_payment DECIMAL(10, 2) NOT NULL,
    credit_limit DECIMAL(10, 2), -- for credit cards
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(user_id)
);

-- Transaction history
CREATE TABLE transactions (
    transaction_id INTEGER PRIMARY KEY AUTOINCREMENT,
    account_id INTEGER,
    transaction_date DATE NOT NULL,
    amount DECIMAL(10, 2) NOT NULL,
    transaction_type VARCHAR(20) NOT NULL, -- payment, charge, interest
    description TEXT,
    balance_after DECIMAL(10, 2),
    FOREIGN KEY (account_id) REFERENCES debt_accounts(account_id)
);

-- Budget categories
CREATE TABLE budget_categories (
    category_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    category_name VARCHAR(50) NOT NULL,
    monthly_budget DECIMAL(10, 2) NOT NULL,
    actual_spending DECIMAL(10, 2) DEFAULT 0,
    FOREIGN KEY (user_id) REFERENCES users(user_id)
);

-- Spending patterns
CREATE TABLE spending_patterns (
    spending_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    category_id INTEGER,
    amount DECIMAL(10, 2) NOT NULL,
    spending_date DATE NOT NULL,
    merchant VARCHAR(100),
    FOREIGN KEY (user_id) REFERENCES users(user_id),
    FOREIGN KEY (category_id) REFERENCES budget_categories(category_id)
);

-- Payoff recommendations
CREATE TABLE payoff_recommendations (
    recommendation_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    strategy_name VARCHAR(50) NOT NULL, -- snowball, avalanche, optimized
    total_interest_saved DECIMAL(10, 2),
    months_saved INTEGER,
    recommended_extra_payment DECIMAL(10, 2),
    priority_order TEXT, -- JSON array of account priorities
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(user_id)
);

-- Create indexes for better performance
CREATE INDEX idx_user_debts ON debt_accounts(user_id);
CREATE INDEX idx_account_transactions ON transactions(account_id);
CREATE INDEX idx_user_spending ON spending_patterns(user_id);
CREATE INDEX idx_spending_date ON spending_patterns(spending_date);
