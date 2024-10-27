import pandas as pd
import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt
import seaborn as sns

# Load the DialoGPT model and tokenizer
model_name = "microsoft/DialoGPT-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Function to detect fraud based on some simple rules
def detect_fraud(transactions):
    flagged_transactions = []
    explanations = []

    # Check for duplicate Transaction IDs
    duplicates = transactions[transactions.duplicated(['Transaction_id'], keep=False)]
    for _, row in duplicates.iterrows():
        flagged_transactions.append(row)
        explanations.append({
            "Status": "Fraud",
            "Reason": f"Transaction ID '{row['Transaction_id']}' is duplicated.",
            "Recommendation": "Investigation required."
        })

    # Loop through each transaction to check for fraud
    for _, row in transactions.iterrows():
        explanations_for_row = []

        # Sample fraud detection logic
        if row['Amount_Sent'] > 1000000:
            flagged_transactions.append(row)
            explanations_for_row.append({
                "Status": "Flagged Fraud",
                "Reason": f"Flagged due to high amount sent: {row['Amount_Sent']}.",
                "Recommendation": "Confirm with Sender."
            })
        if row['Receiver_type'] == 'Unknown':
            flagged_transactions.append(row)
            explanations_for_row.append({
                "Status": "Flagged Fraud",
                "Reason": "Flagged due to unknown receiver type.",
                "Recommendation": "Confirm with Sender."
            })
        if row['Receiver_location'] == 'Unknown':
            flagged_transactions.append(row)
            explanations_for_row.append({
                "Status": "Fraud",
                "Reason": "Flagged due to unknown receiver location.",
                "Recommendation": "Investigation required."
            })

        # Check for transactions on the same day for the same user
        transactions_on_same_day = transactions[transactions['Date'] == row['Date']]
        user_transactions = transactions_on_same_day[transactions_on_same_day['Customer_name'] == row['Customer_name']]
        if len(user_transactions) > 5:
            flagged_transactions.append(row)
            explanations_for_row.append({
                "Status": "Flagged Fraud",
                "Reason": f"{row['Customer_name']} has more than 5 transactions on the same day.",
                "Recommendation": "Confirm with Sender."
            })

        # Append explanations for the current row to the main explanations list
        explanations.extend(explanations_for_row)

    return pd.DataFrame(flagged_transactions), explanations

# Function to create a report
def create_report(flagged_df, explanations):
    report_data = []
    explanation_index = 0  # Track explanation index

    for index, row in flagged_df.iterrows():
        # Get explanation for the current transaction
        if explanation_index < len(explanations):
            explanation = explanations[explanation_index]
            explanation_index += 1
        else:
            explanation = {
                "Status": "Flagged Fraud",
                "Reason": "No explanation available.",
                "Recommendation": "Investigate further."
            }

        report_entry = {
            "Date": row['Date'],
            "Customer_name": row['Customer_name'],
            "Transaction_id": row['Transaction_id'],
            "Amount_Sent": row['Amount_Sent'],
            "Receiver_type": row['Receiver_type'],
            "Receiver_location": row['Receiver_location'],
            "Status": explanation["Status"],
            "Reason": explanation["Reason"],
            "Recommendation": explanation["Recommendation"]
        }
        report_data.append(report_entry)

    return pd.DataFrame(report_data)

# Function to calculate fraud statistics
def fraud_statistics(flagged_df, total_transactions):
    total_flagged = len(flagged_df)
    fraud_percentage = (total_flagged / total_transactions) * 100 if total_transactions > 0 else 0
    return {
        "Total Flagged Transactions": total_flagged,
        "Fraud Percentage": fraud_percentage
    }

# Function to display visualizations
def display_visualizations(transactions):
    # Visualizing receiver types
    plt.figure(figsize=(10, 6))
    sns.countplot(data=transactions, x='Receiver_location', order=transactions['Receiver_location'].value_counts().index)
    plt.title('Count of Transactions by Receiver Locations')
    plt.xticks(rotation=45)
    st.pyplot(plt)

# Function to filter flagged transactions
def filter_flagged_transactions(flagged_df):
    min_amount = st.number_input("Minimum Amount", min_value=0, value=0)
    max_amount = st.number_input("Maximum Amount", value=1000000)
    
    filtered_df = flagged_df[(flagged_df['Amount_Sent'] >= min_amount) & (flagged_df['Amount_Sent'] <= max_amount)]
    return filtered_df

# Function to display DataFrame with row count
def display_dataframe_with_count(df, title):
    st.write(f"{title} (Total Rows: {len(df)})")
    st.dataframe(df)

# Streamlit application
st.subheader("Transaction Fraud Detection Chatbot 🤖")

# Upload CSV
uploaded_file = st.file_uploader("Upload your transactions CSV file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Fill NaN values with 'Unknown'
    df.fillna('Unknown', inplace=True)

    # Display uploaded data with row count
    display_dataframe_with_count(df, "Uploaded Transactions")

    # Fraud detection
    flagged_df, explanations = detect_fraud(df)
    if not flagged_df.empty:
        display_dataframe_with_count(flagged_df, "Flagged Transactions for Fraud")
        
        # Display fraud statistics
        stats = fraud_statistics(flagged_df, len(df))
        st.write("Fraud Statistics:")
        for key, value in stats.items():
            st.write(f"{key}: {value:.2f}")

        # Option to filter flagged transactions
        st.subheader("Filter Flagged Transactions")
        filtered_flagged_df = filter_flagged_transactions(flagged_df)
        display_dataframe_with_count(filtered_flagged_df, "Filtered Flagged Transactions")

        # Visualizations
        st.subheader("Visualizations")
        display_visualizations(flagged_df)
        
    else:
        st.write("No fraudulent transactions detected.")

    # Option to download raw data
    st.download_button(
        label="Download Raw Data",
        data=df.to_csv(index=False),
        file_name='transactions.csv',
        mime='text/csv'
    )

    # Initialize chat state
    if 'responses' not in st.session_state:
        st.session_state['responses'] = []
    if 'requests' not in st.session_state:
        st.session_state['requests'] = []

    # Container for chat history
    response_container = st.container()
    textcontainer = st.container()

    # Text input for user query
    with textcontainer:
        query = st.text_input("Ask a question about the transactions: ", key="input")
        if query:
            with st.spinner("Generating response..."):
                # Check for report requests
                if "report" in query.lower():
                    report_df = create_report(flagged_df, explanations)
                    report_csv = report_df.to_csv(index=False)

                    # Provide options to download the report
                    st.download_button(
                        label="Download CSV Report",
                        data=report_csv,
                        file_name='fraud_report.csv',
                        mime='text/csv'
                    )
                    
                    response = "The Report has been generated. You can download it using the buttons below ⬇️."
                    st.session_state.requests.append(query)
                    st.session_state.responses.append(response)
                else:
                    # Create a conversation context
                    context = flagged_df.to_string(index=False)  # Convert flagged transactions to string
                    context = context[:2000]  # Limit context to the first 2000 characters
                    full_input = f"Context:\n{context}\n\nUser Query:\n{query}"

                    # Tokenize the input
                    inputs = tokenizer.encode(full_input + tokenizer.eos_token, return_tensors='pt')

                    # Ensure the input length does not exceed the model's capacity
                    if inputs.size(1) > 2048:  # DialoGPT's max input length is 2048 tokens
                        inputs = inputs[:, -2048:]  # Use the last 2048 tokens

                    # Generate a response
                    response_ids = model.generate(inputs, max_length=2000, num_return_sequences=1)

                    # Check if response_ids is not empty before accessing
                    if response_ids.size(0) > 0:
                        response = tokenizer.decode(response_ids[0], skip_special_tokens=True)
                    else:
                        response = "I'm sorry, I couldn't generate a response."
                    st.session_state.requests.append(query)
                    st.session_state.responses.append(response)

    with response_container:
        if st.session_state['responses']:
            for i in range(len(st.session_state['responses'])):
                st.write(f"User: {st.session_state['requests'][i]}")
                st.write(f"Bot: {st.session_state['responses'][i]}")
