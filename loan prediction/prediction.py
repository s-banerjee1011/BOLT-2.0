import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score
import warnings

warnings.filterwarnings("ignore")

class LoanPredictionPlatform:
    def __init__(self, data_path="C:\\GENERAL\\lp\\loan_prediction_dataset_shuffled.csv"):
        self.data_path = data_path
        self.df = self.load_data()
        self.features = ['Liquidity', 'Company Current Funds', 'Frequency of taking loans',
                         'Credit History', 'Profit Percentage', 'Liquidity to Funds Ratio',
                         'Credit-Profit Interaction', 'Liquidity-Funds Interaction',
                         'Credit*Frequency', 'Profit^2']
        self.target = 'Loan Amount to be Required'
        self.model = None

    def load_data(self):
        if os.path.exists(self.data_path):
            df = pd.read_csv(self.data_path)
            if 'Interest Rate' in df.columns:
                df['Interest Rate'] = df['Interest Rate'].astype(str).str.replace('%', '').astype(float) / 100

            df['Liquidity to Funds Ratio'] = df['Liquidity'] / (df['Company Current Funds'] + 1)
            df['Credit-Profit Interaction'] = df['Credit History'] * df['Profit Percentage']
            df['Liquidity-Funds Interaction'] = df['Liquidity'] * df['Company Current Funds']
            df['Credit*Frequency'] = df['Credit History'] * df['Frequency of taking loans']
            df['Profit^2'] = df['Profit Percentage'] ** 2
            return df
        else:
            columns = ['Company Name', 'Liquidity', 'Company Current Funds', 'Frequency of taking loans',
                       'Loan Id', 'Loan Amount Term', 'Credit History', 'Interest Rate', 'Profit Percentage',
                       'Company to require Loan', 'Loan Amount to be Required', 'Suitable Company to provide Loan',
                       'Amount to refund after 5 yrs', 'Liquidity to Funds Ratio', 'Credit-Profit Interaction',
                       'Liquidity-Funds Interaction', 'Credit*Frequency', 'Profit^2']
            return pd.DataFrame(columns=columns)

    def save_data(self):
        self.df.to_csv(self.data_path, index=False)

    def add_new_company(self):
        print("\nAdd a New Company Entry")
        company = input("Company Name: ")
        liquidity = float(input("Liquidity (₹): "))
        funds = float(input("Current Funds (₹): "))
        frequency = int(input("Loan Frequency (0–5): "))
        loan_id = input("Loan ID (e.g., LN0001): ")
        term = int(input("Loan Term (months): "))
        credit = float(input("Credit History (300–850): "))
        interest = float(input("Interest Rate (9–12%): ")) / 100
        profit = float(input("Profit Percentage (-20 to 50): "))

        liquidity_to_funds = liquidity / (funds + 1)
        credit_profit = credit * profit
        liquidity_funds = liquidity * funds
        credit_freq = credit * frequency
        profit_sq = profit ** 2

        needs_loan = self.predict_loan_requirement(liquidity, credit, frequency)
        loan_amount = 0
        suitable_lenders = ""
        refund_amount = 0

        if needs_loan:
            loan_amount = self.calculate_loan_amount(liquidity, funds, profit)
            suitable_lenders = self.find_suitable_lenders(loan_amount, credit)
            refund_amount = loan_amount * ((1 + interest) ** (term / 12))

        row = {
            'Company Name': company,
            'Liquidity': liquidity,
            'Company Current Funds': funds,
            'Frequency of taking loans': frequency,
            'Loan Id': loan_id,
            'Loan Amount Term': term,
            'Credit History': credit,
            'Interest Rate': interest,
            'Profit Percentage': profit,
            'Company to require Loan': needs_loan,
            'Loan Amount to be Required': loan_amount if needs_loan else np.nan,
            'Suitable Company to provide Loan': suitable_lenders if needs_loan else "",
            'Amount to refund after 5 yrs': refund_amount if needs_loan else np.nan,
            'Liquidity to Funds Ratio': liquidity_to_funds,
            'Credit-Profit Interaction': credit_profit,
            'Liquidity-Funds Interaction': liquidity_funds,
            'Credit*Frequency': credit_freq,
            'Profit^2': profit_sq
        }

        self.df = pd.concat([self.df, pd.DataFrame([row])], ignore_index=True)
        self.save_data()
        print("\nCompany Added Successfully!")
        
        # Show the company most likely to require a loan
        self.show_most_likely_loan()

    def show_most_likely_loan(self):
        # Identify the company most likely to need a loan based on the dataset
        loan_df = self.df[self.df['Company to require Loan'] == True]
        if loan_df.empty:
            print("No companies are predicted to require a loan in the dataset.")
            return
        
        most_likely_company = loan_df.loc[loan_df['Loan Amount to be Required'].idxmax()]
        
        print("\nThe company most likely to require a loan in the near future is: ", most_likely_company['Company Name'])
        print(f"Predicted Loan Amount: ₹{most_likely_company['Loan Amount to be Required']:,.2f}")
        print(f"Suggested Lenders: {most_likely_company['Suitable Company to provide Loan']}")
        print(f"Estimated Repayment after 5 years: ₹{most_likely_company['Amount to refund after 5 yrs']:,.2f}")

    def predict_loan_requirement(self, liquidity, credit, frequency):
        liquidity_score = max(0, 1 - liquidity / 5000000)
        credit_score = credit / 850
        frequency_score = frequency / 5
        loan_prob = 0.5 * liquidity_score + 0.3 * (1 - credit_score) + 0.2 * frequency_score
        return loan_prob > 0.5

    def calculate_loan_amount(self, liquidity, funds, profit):
        if profit < 0:
            base = max(100000, (1000000 - liquidity) * 1.5)
        else:
            base = max(100000, (1000000 - liquidity) * (1 + profit / 100))
        return min(base, 10000000)

    def find_suitable_lenders(self, loan_amount, min_credit_score=600):
        lenders = self.df[(self.df['Company Current Funds'] > loan_amount * 1.5) & 
                          (self.df['Credit History'] >= min_credit_score) & 
                          (self.df['Company to require Loan'] == False)].copy()

        if lenders.empty:
            return "No suitable lenders found"

        lenders['Financial Capacity'] = self.calculate_financial_capacity(lenders)
        top_lenders = lenders.sort_values('Financial Capacity', ascending=False).head(3)

        total_capacity = top_lenders['Company Current Funds'] * 0.3
        total_capacity_sum = total_capacity.sum()
        adjusted_portions = (total_capacity / total_capacity_sum) * loan_amount

        result = []
        for (idx, row), portion in zip(top_lenders.iterrows(), adjusted_portions):
            result.append(f"{row['Company Name']} - ₹{portion:,.2f}")

        return "    ".join(result)

    def calculate_financial_capacity(self, df):
        liquidity = df['Liquidity'] / df['Liquidity'].max()
        funds = df['Company Current Funds'] / df['Company Current Funds'].max()
        credit = df['Credit History'] / 850
        profit = (df['Profit Percentage'] + 20) / 70
        return 0.2 * liquidity + 0.3 * funds + 0.3 * credit + 0.2 * profit

    def prepare_data(self):
        loan_df = self.df[self.df['Company to require Loan'] == True].copy()
        if loan_df.shape[0] < 10:
            print("Not enough loan-requiring companies to train the model.")
            return False

        # Clip outliers
        for col in self.features:
            loan_df[col] = loan_df[col].clip(lower=loan_df[col].quantile(0.01),
                                             upper=loan_df[col].quantile(0.99))

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            loan_df[self.features], loan_df[self.target], test_size=0.2, random_state=42)
        return True

    def build_model(self):
        num_pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('poly', PolynomialFeatures(degree=2, include_bias=False)),
            ('pca', PCA(n_components=0.98))
        ])

        preprocessor = ColumnTransformer([
            ('num', num_pipe, self.features)
        ])

        pipe = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', GradientBoostingRegressor(random_state=42))
        ])

        grid = {
            'regressor__n_estimators': [200, 300],
            'regressor__learning_rate': [0.05, 0.1],
            'regressor__max_depth': [4, 5],
            'regressor__min_samples_split': [2, 5],
            'regressor__min_samples_leaf': [1, 3],
            'regressor__subsample': [0.8, 1.0],
            'regressor__max_features': ['sqrt']
        }

        self.model = GridSearchCV(pipe, grid, cv=5, scoring='r2', n_jobs=-1)
        self.model.fit(self.X_train, self.y_train)

    def predict_loan_amount(self):
        print("\nPredict Loan for a New Company")
        liquidity = float(input("Liquidity (₹): "))
        funds = float(input("Current Funds (₹): "))
        frequency = int(input("Loan Frequency (0–5): "))
        credit = float(input("Credit Score (300–850): "))
        profit = float(input("Profit Percentage (-20 to 50): "))

        liquidity_to_funds = liquidity / (funds + 1)
        credit_profit = credit * profit
        liquidity_funds = liquidity * funds
        credit_freq = credit * frequency
        profit_sq = profit ** 2

        new_df = pd.DataFrame([[liquidity, funds, frequency, credit, profit,
                                liquidity_to_funds, credit_profit,
                                liquidity_funds, credit_freq, profit_sq]],
                              columns=self.features)

        if self.model is None:
            if self.prepare_data():
                self.build_model()
            else:
                return

        amount = self.model.predict(new_df)[0]
        print(f"Predicted Loan Amount: ₹{amount:,.2f}")

        if self.predict_loan_requirement(liquidity, credit, frequency):
            lenders = self.find_suitable_lenders(amount)
            interest = 0.09 + 0.03 * (1 - credit / 850)
            repay = amount * (1 + interest) ** 5
            print(f"Suggested Lenders: {lenders}")
            print(f"Estimated Repayment (5 yrs): ₹{repay:,.2f}")
        else:
            print("Company likely doesn't need a loan.")

    def run(self):
        while True:
            print("\n--- Business Loan Platform ---")
            print("1. Add a New Company")
            print("2. Predict Loan for a New Company")
            print("3. Exit")
            choice = input("Choose (1/2/3): ")
            if choice == '1':
                self.add_new_company()
            elif choice == '2':
                self.predict_loan_amount()
            elif choice == '3':
                print("Goodbye!")
                break
            else:
                print("Invalid option. Try again.")

if __name__ == "__main__":
    LoanPredictionPlatform().run()
