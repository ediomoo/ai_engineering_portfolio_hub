import streamlit as st
import streamlit.components.v1 as components
from src.loan.pipeline.predict_pipeline import CustomData, PredictPipeline

def run_loan_ui():
    st.header("💰 Loan Approval Prediction")

    tab1, tab2 = st.tabs(["Launch App", "🔬 View Data Research"])

    with tab1:
        
        st.markdown("Enter the applicant's details below to check loan eligibility.")

        # st.cache_resource so the model doesnt reload when switching pages
        @st.cache_resource
        def load_loan_pipeline():
            return PredictPipeline()

        pipeline = load_loan_pipeline()

        # --- User Input Form ---
        with st.form("loan_form"):

            col1, col2 = st.columns(2)

            with col1:
                gender = st.selectbox("Gender", ["Male", "Female"])
                married = st.selectbox("Married", ['Yes', 'No'])
                # ADDED: Dependents input
                dependents = st.selectbox("Dependents", ['0', '1', '2', '3+']) 
                education = st.selectbox("Education", ['Graduate', 'Not Graduate'])
                applicant_income = st.number_input("Applicant Income ($)", min_value=0.0, value=5000.0, step=100.0)
                loan_amount = st.number_input("Loan Amount ($k)", min_value=0.0, value=150.0, step=10.0)

            with col2:
                self_employed = st.selectbox("Self Employed", ['No', 'Yes'])
                property_area = st.selectbox("Property Area", ['Rural', 'Urban', 'Semiurban'])
                credit_history = st.selectbox("Credit History", [1.0, 0.0], help="1.0 for good history, 0.0 for poor")
                coapplicant_income = st.number_input("Co-applicant Income ($)", min_value=0.0, value=0.0, step=100.0)
                loan_amount_term = st.number_input("Loan Term (Days)", min_value=0.0, value=360.0, step=30.0)

        
            # Submit button for the form
            submit = st.form_submit_button("Predict Approval Status")


        # --- Prediction Logic ---
        if submit:
            try:
                with st.spinner('Analyzing application...'):
                
                    data_obj = CustomData(
                        Gender=gender, 
                        Married=married, 
                        Dependents=dependents, # <--- Added this
                        Education=education, 
                        Self_Employed=self_employed,
                        ApplicantIncome=applicant_income, 
                        CoapplicantIncome=coapplicant_income, 
                        LoanAmount=loan_amount, 
                        Loan_Amount_Term=loan_amount_term, 
                        Credit_History=credit_history,
                        Property_Area=property_area
                    )

                    # 2. Convert dictionary to DataFrame for the preprocessor 
                    df = data_obj.get_as_df()

                    # 3. Generate Prediction
                    result = pipeline.predict(df)
            
                    # 4. Format and display output
                    status = "Approved" if result[0] == 1 else "Rejected"

                    st.markdown("---")
                    if status == "Approved":
                        st.success(f"### Result: **{status}** ✅")
                        st.balloons()
                    else:
                        st.error(f"### Result: **{status}** ❌")

            except Exception as e:
                st.error(f'An error occured during prediction: {e}')
    
    with tab2:

        st.subheader("Exploratory Data Analysis Report")
        st.write("This report shows the statistical evidence used to build the model.")
        
        # Load the HTML file we converted in Step 1
        with open("notebooks/loan/exploration.html", 'r', encoding='utf-8') as f:
            html_data = f.read()
        
        # Display the notebook as an embedded frame
        components.html(html_data, height=800, scrolling=True)


# Call the function (for testing individually)
if __name__ == "__main__":
    run_loan_ui()




