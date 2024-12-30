import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import plotly.express as px
import io

# Streamlit Page Configuration
st.set_page_config(
    page_title="Fraud Detection App",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
        body {
            background-color: #f7f9fc;
        }
        .stButton>button {
            color: white;
            background-color: #4CAF50;
            border-radius: 5px;
            padding: 10px 20px;
        }
    </style>
""", unsafe_allow_html=True)

# Store user credentials (you can replace this with a database in a real app)
USER_CSV = "C:\\Users\\prart\\Desktop\\1\\Creditcard\\random_user_data.csv"


# Function to load existing users from the CSV
def load_users():
    try:
        return pd.read_csv(USER_CSV)
    except FileNotFoundError:
        return pd.DataFrame(columns=["username", "password"])

# Function to save users to the CSV (without hashing)
def save_user(username, password):
    user_data = pd.DataFrame({"username": [username], "password": [password]})
    user_data.to_csv(USER_CSV, mode='a', header=False, index=False)

# Function to check password (direct comparison)
def check_password(entered_password, stored_password):
    return entered_password == stored_password

# Sign-Up page
def sign_up():
    st.title("🔑 Create a New Account:Sign-Up")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    confirm_password = st.text_input("Confirm Password", type="password")
    
    if password != confirm_password:
        st.error("🚨 Passwords do not match!")
    elif st.button("Sign Up"):
        # Check if the username already exists
        existing_users = load_users()
        if username in existing_users["username"].values:
            st.error("🚨 Username already exists!")
        else:
            # Save the new user credentials without hashing
            save_user(username, password)
            st.success("🎉Account created successfully! You can now log in.")

# Login page
def login():
    st.title("🔒 Login to Your Account")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    
    if st.button("Login"):
        # Check if the user exists
        existing_users = load_users()
        if username not in existing_users["username"].values:
            st.error("🚨 Username not found!")
        else:
            # Get the stored password from CSV
            stored_password = existing_users.loc[existing_users["username"] == username, "password"].values[0]
            
            # Compare the entered password with the stored password (no hashing)
            if check_password(password, stored_password):
                st.success(f"🎉Welcome back, {username}!")
                # Set login state in session
                st.session_state.logged_in = True
                st.session_state.username = username
                # Proceed to the next part of the app (fraud detection logic)
                st.session_state.page = "fraud_detection"
            else:
                st.error("❌Incorrect password!")

# Fraud Detection Logic (App)
def fraud_detection_app():
    if 'logged_in' not in st.session_state or not st.session_state.logged_in:
        st.warning("🔔You need to log in first.")
        return

    st.title("🚨--Fraud Detection Dashboard--🚨")

    # Sidebar for data upload
    st.sidebar.header("Upload Dataset")
    uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type="csv")

    if uploaded_file:
        # Load dataset
        data = pd.read_csv(uploaded_file)
        st.success(f"✅Dataset loaded with {data.shape[0]} rows and {data.shape[1]} columns.")
        
        # Display dataset overview
        st.write("### Dataset Overview")
        st.write(data.head())
        st.write(f"The dataset has {data.shape[0]} rows and {data.shape[1]} columns.")
        
        # Option to choose between Credit Card or UPI
        transaction_type = st.selectbox("Select Transaction Type", ["Credit Card", "UPI"])

        if transaction_type == "Credit Card":
            # Fraud Detection Logic for Credit Cards
            st.write("### Credit Card Fraud Detection")
            
            # 1. Detect outliers in Transaction Amounts using Isolation Forest
            st.write("### Detecting Outliers in Transaction Amounts")
            isolation_forest = IsolationForest(contamination=0.01)
            outlier_predictions = isolation_forest.fit_predict(data[['Amount']])
            data['Outlier'] = outlier_predictions
            outliers = data[data['Outlier'] == -1]
            
            st.write(f"Detected {outliers.shape[0]} potential outliers in transaction amounts.")
            if not outliers.empty:
                st.write("### Outliers (Potential Fraudulent Transactions)")
                st.write(outliers[['Transaction ID', 'Card Number', 'Amount', 'Class']])

            # Save Model Option
            if st.button("💳Save Credit Card Model"):
                joblib.dump(isolation_forest, "credit_card_isolation_forest_model.pkl")
                st.success("Credit Card Model saved successfully as 'credit_card_isolation_forest_model.pkl'!")

            # 2. Detect Suspicious Frequency of Transactions
            st.write("### Detecting Suspicious Frequency of Transactions")
            transaction_counts = data.groupby('Card Number')['Transaction ID'].count()
            suspicious_cards = transaction_counts[transaction_counts > transaction_counts.quantile(0.95)]  # Cards with transactions in top 5%
            
            st.write(f"Detected {suspicious_cards.shape[0]} cards with suspicious transaction frequency.")
            if not suspicious_cards.empty:
                st.write("### Suspicious Cards Based on Frequency")
                st.write(suspicious_cards)

            # 3. Flag Cards with High Fraud Activity
            st.write("### Flagging Cards with High Fraud Activity")
            fraud_counts = data[data['Class'] == 1].groupby('Card Number')['Class'].count()
            card_activity = transaction_counts.to_frame().join(fraud_counts.to_frame(), how='left')
            card_activity.columns = ['Transaction Count', 'Fraud Count']
            card_activity['Fraud Count'] = card_activity['Fraud Count'].fillna(0)
            sorted_card_activity = card_activity.sort_values(by='Fraud Count', ascending=False)
            
            st.write(f"Detected {sorted_card_activity.shape[0]} cards with fraud activity, sorted from most fraudulent to least.")
            st.write("### Cards Sorted by Fraud Activity")
            st.write(sorted_card_activity)

        elif transaction_type == "UPI":
    # Fraud Detection Logic for UPI
         st.write("### UPI Fraud Detection")

    # 4. Detect Suspicious UPI Transaction Frequency
         st.write("### Detecting Suspicious UPI Transaction Frequency")

    # Check if the 'Transaction Method' column exists in the dataset
         if 'Transaction Method' in data.columns:
             upi_data = data[data['Transaction Method'] == 'UPI']
             upi_counts = upi_data.groupby('Transaction ID').count()
             suspicious_upi_cards = upi_counts[upi_counts > upi_counts.quantile(0.95)]  # UPI transactions in top 5%

           
             st.write("### Comparing Transaction ID with Fraud Count")
             transaction_id_input = st.text_input("Enter Transaction ID to check for fraud")
        
             if st.button("Check Fraud for Transaction ID"):
                 if transaction_id_input:
                # Ensure the transaction ID exists in the dataset
                  if transaction_id_input in upi_data['Transaction ID'].values:
                    # Check if the transaction is fraudulent (Class == 1)
                      fraud_count = upi_data[upi_data['Transaction ID'] == transaction_id_input]['Class'].sum()
                      if fraud_count > 0:
                          st.error(f"🚨 Transaction ID {transaction_id_input} is fraudulent! Fraud count: {fraud_count}")
                      else:
                          st.success(f"🎉Transaction ID {transaction_id_input} is legitimate. Fraud count: {fraud_count}")
                  else:
                    st.warning("🚨 Transaction ID not found in the UPI dataset.")
    

        # Save Model Option for UPI
             if st.button("💾Save UPI Model"):
                 joblib.dump(upi_counts, "upi_transaction_model.pkl")
                 st.success("✅UPI model saved successfully as 'upi_transaction_model.pkl'!")
         else:
             st.error("🚨 'Transaction Method' column is missing from the dataset.")

   # New Section: Transaction Validation
         st.write("### Validate Transaction by ID")
         transaction_id = st.text_input("Enter Transaction ID to Validate")
         transaction_amount = st.text_input("Enter Transaction Amount to Validate")


         if st.button("Validate Transaction"):
             try:
        # Convert the input to match the data type of the Transaction ID column
                if 'Transaction ID' in data.columns:
            # Ensure type consistency
                    column_dtype = data['Transaction ID'].dtype
                    if column_dtype == 'int64':  # If Transaction IDs are integers
                        transaction_id = int(transaction_id)
                    elif column_dtype == 'float64':  # If Transaction IDs are floats
                        transaction_id = float(transaction_id)
            
                    # Validate and process Transaction Amount
                    column_dtype_amount = data['Amount'].dtype
                    if column_dtype_amount == 'int64':  # If Amounts are integers
                        transaction_amount = int(transaction_amount)
                    elif column_dtype_amount == 'float64':  # If Amounts are floats
                        transaction_amount = float(transaction_amount)

                    # Check if the Transaction ID exists in the dataset
                    if transaction_id in data['Transaction ID'].values:
                        transaction_status = data.loc[data['Transaction ID'] == transaction_id, 'Class'].values[0]
                
                        if transaction_status == 1:  # Fraudulent transaction
                            st.error(f"🚨 Transaction ID {transaction_id} is marked as fraudulent. Please take action!")
                        else:  # Legitimate transaction
                            st.success(f"✅Transaction ID {transaction_id} is legitimate.")
                
                # Check if the amount matches the dataset

                    # Verify if the provided amount matches
                        stored_amount = data.loc[data['Transaction ID'] == transaction_id, 'Amount'].values[0]
                        if transaction_amount == stored_amount:
                            st.success(f"The entered amount {transaction_amount} matches the recorded amount.✅")
                        else:
                            st.warning(f"⚠️The entered amount {transaction_amount} does not match the recorded amount {stored_amount}. Proceed with caution.")
                    else:
                        st.warning("⚠️Transaction ID not found in the dataset. Please verify and try again.")
                else:
                    st.error("⚠️'Transaction ID' or 'Amount' column is missing from the dataset.")
             except ValueError:
                 st.error("❌Invalid input. Please ensure both Transaction ID and Amount match the expected format.")
    # Add Machine Learning Model Selection
    model_type = st.selectbox("Select Machine Learning Model", ["Logistic Regression", "Random Forest", "Decision Tree"])

    if st.button("Train Model"):
        try:
            st.write(f"Training {model_type} model...")

            # Ensure the 'Class' column exists
            if 'Class' not in data.columns:
                st.error("The dataset must contain a 'Class' column.")
                return

            X = data.drop(['Class'], axis=1)  # Features
            y = data['Class']  # Target variable

            # Check if the dataset has enough rows and columns for training
            if X.shape[0] < 10:
                st.error("⚠️Not enough data for training!")
                return

            # Handle non-numeric columns by encoding them
            for column in X.select_dtypes(include=['object']).columns:
                le = LabelEncoder()
                X[column] = le.fit_transform(X[column])

            if model_type == "Logistic Regression":
                model = LogisticRegression(max_iter=1000)
            elif model_type == "Random Forest":
                model = RandomForestClassifier()
            elif model_type == "Decision Tree":
                model = DecisionTreeClassifier()

            # Train the model
            model.fit(X, y)

            # Save the model
            model_filename = f"{model_type.replace(' ', '_').lower()}_model.pkl"
            joblib.dump(model, model_filename)
            st.success(f"{model_type} model trained and saved as '{model_filename}'!")

            # Predictions
            y_pred = model.predict(X)

            # Confusion Matrix with Plotly Heatmap
            cm = confusion_matrix(y, y_pred)
            cm_df = pd.DataFrame(cm, columns=["Not Fraud", "Fraud"], index=["Not Fraud", "Fraud"])
            fig = px.imshow(cm_df, 
                            labels=dict(x="Predicted", y="True", color="Counts"), 
                            x=["Not Fraud", "Fraud"], 
                            y=["Not Fraud", "Fraud"], 
                            color_continuous_scale="Blues")
            st.plotly_chart(fig)

            # Classification Report as DataFrame
            report = classification_report(y, y_pred, output_dict=True)
            report_df = pd.DataFrame(report).transpose()

            # Display classification report interactively
            st.write("### Classification Report")
            st.dataframe(report_df)  # Interactive table

            # Button to download the classification report as CSV
            csv = report_df.to_csv(index=False)
            st.download_button(
                label="📥Download Classification Report",
                data=csv,
                file_name="classification_report.csv",
                mime="text/csv"
            )

        except Exception as e:
            st.error(f"Error during training: {e}")

# Main app logic
def main():
    # Initialize session state if not already done
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
        st.session_state.page = "login"

    menu = ["Sign Up", "Login"]
    choice = st.sidebar.radio("Choose an option:", ["Sign Up", "Login"])
    
    if choice == "Sign Up":
        sign_up()
    elif choice == "Login":
        login()
    
    if st.session_state.page == "fraud_detection":
        fraud_detection_app()

# Run the app
if __name__ == "__main__":
    main()