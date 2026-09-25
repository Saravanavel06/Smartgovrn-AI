import streamlit as st
import joblib
import pandas as pd
import re

st.set_page_config(
    page_title="SmartGov AI",
    page_icon="🤖"
)

st.title("🤖 SmartGov AI")
st.subheader("E-Governance Assistant")

st.write(
    "Welcome to SmartGov AI! "
    "Ask questions about government services."
)

print("SmartGov AI application started successfully!")
# ==============================
# LOAD TRAINED MODEL
# ==============================

model = joblib.load(
    "models/logistic_regression_model.pkl"
)

# Load TF-IDF vectorizer
tfidf = joblib.load(
    "models/tfidf_vectorizer.pkl"
)

# Load application status data
status_df = pd.read_csv(
    "data/mock_application_status.csv"
)

print("Model loaded successfully!")
print("TF-IDF vectorizer loaded successfully!")
print("Status data loaded successfully!")
# ==============================
# TEXT CLEANING
# ==============================

def clean_text(text):

    text = text.lower()

    text = re.sub(
        r'[^a-zA-Z0-9\s]',
        '',
        text
    )

    text = re.sub(
        r'\s+',
        ' ',
        text
    ).strip()

    return text


# ==============================
# RESPONSE ENGINE
# ==============================

responses = {

    "greeting":
        "Hello! I am SmartGov AI. I can help you with passport, birth certificate, ration card, driving licence, and income certificate services.",

    "goodbye":
        "Thank you for using SmartGov AI. Have a great day!",

    "passport_application":
        "For passport application, you need to complete the application process and submit the required documents through the official passport service portal.",

    "passport_documents":
        "Common passport application documents include identity proof, address proof, and other documents depending on your application type. Please check the official passport portal for the current requirements.",

    "passport_renewal":
        "For passport renewal, you need to submit a renewal application along with the required documents. Please check the official passport service portal for current procedures.",

    "birth_certificate_application":
        "To apply for a birth certificate, submit the required birth details and supporting documents through the appropriate government service.",

    "birth_certificate_documents":
        "Birth certificate applications generally require birth-related details and supporting documents. Please check the official government service for the current document requirements.",

    "birth_certificate_correction":
        "For correcting details in a birth certificate, submit a correction request along with supporting documents proving the correct information.",

    "ration_card_application":
        "To apply for a ration card, submit the required family and address details through the appropriate government service.",

    "ration_card_documents":
        "Ration card applications generally require identity proof, address proof, and family-related details.",

    "ration_card_status":
        "You can check your ration card application status using the application or reference number provided during registration.",

    "driving_license_application":
        "To apply for a driving licence, complete the required application process and submit the necessary documents.",

    "driving_license_documents":
        "Driving licence applications generally require identity proof, address proof, age proof, and other documents depending on the application type.",

    "driving_license_renewal":
        "For driving licence renewal, submit a renewal application along with the required documents before or after the licence expiry as applicable.",

    "income_certificate_application":
        "To apply for an income certificate, provide the required personal and income-related information through the appropriate government service.",

    "income_certificate_documents":
        "Income certificate applications may require identity proof, address proof, and income-related supporting documents.",

    "income_certificate_status":
        "You can check your income certificate application status using the application or reference number.",

    "out_of_scope":
        "Sorry, I can currently help only with the supported e-governance services such as passport, birth certificate, ration card, driving licence, and income certificate."
}# ==============================
# APPLICATION STATUS FUNCTION
# ==============================

def check_application_status(application_id):

    application_id = application_id.strip().upper()

    result = status_df[
        status_df["application_id"] == application_id
    ]

    if result.empty:

        return (
            "Application ID not found. "
            "Please check the ID and try again."
        )

    row = result.iloc[0]

    return (
        f"Application ID: {row['application_id']}\n\n"
        f"Service: {row['service']}\n\n"
        f"Status: {row['status']}"
    )


# ==============================
# CHATBOT PREDICTION FUNCTION
# ==============================

def smartgov_chat(user_input):

    user_input = user_input.strip()

    # Check if user entered an application ID
    if user_input.upper().startswith("DEMO"):

        return check_application_status(user_input)

    # Clean user question
    cleaned_question = clean_text(user_input)

    # Convert question to TF-IDF
    question_tfidf = tfidf.transform(
        [cleaned_question]
    )

    # Predict intent using trained model
    predicted_intent = model.predict(
        question_tfidf
    )[0]

    # Get response
    response = responses.get(
        predicted_intent,
        "Sorry, I could not understand your question."
    )

    return response
    # ==============================
# CHATBOT USER INTERFACE
# ==============================

st.divider()

st.write("### 💬 Ask SmartGov AI")

user_input = st.text_input(
    "Enter your question or Application ID:",
    placeholder="Example: How can I apply for a passport?"
)

if st.button("Send"):

    if user_input.strip():

        response = smartgov_chat(user_input)

        st.success("🤖 SmartGov AI")
        st.write(response)

    else:

        st.warning(
            "Please enter a question or Application ID."
        )


st.divider()

st.caption(
    "SmartGov AI is an academic demonstration chatbot. "
    "Application status shown here uses fictional demo data."
)