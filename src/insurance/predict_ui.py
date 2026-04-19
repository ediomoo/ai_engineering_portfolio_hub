import streamlit as st
import streamlit.components.v1 as components
from src.insurance.pipeline.predict_pipeline import CustomData, PredictPipeline

def run_insurance_ui():
    st.header("🏥 Medical Insurance Cost Prediction")
    
    tab1, tab2 = st.tabs(["Launch App", "🔬 View Data Research"])
    
    with tab1:
        
        st.markdown("Provide individual details to estimate annual medical insurance charges.")



        # st.cache_resource so the model doesnt reload when switching pages
        @st.cache_resource
        def load_insurance_pipeline():
            return PredictPipeline()

        pipeline = load_insurance_pipeline()

        # --- User Input Form ---
        with st.form("insurance_form"):
            col1, col2 = st.columns(2)
        
            with col1:
                # Demographics and basic health
                age = st.number_input("Age", min_value=18, max_value=100, value=30, step=1)
                sex = st.selectbox("Sex", ["male", "female"])
                bmi = st.number_input("Body Mass Index (BMI)", min_value=10.0, max_value=60.0, value=25.0, step=0.1)
        
            with col2:
                # Lifestyle and regional factors
                children = st.number_input("Number of Children", min_value=0, max_value=10, value=0)
                smoker = st.selectbox("Smoker Status", ["no", "yes"])
                region = st.selectbox("Region", ["southwest", "southeast", "northwest", "northeast"])

        

            # Predict button
            submit = st.form_submit_button("Estimate Insurance Cost")


        # --- Prediction Logic ---
        if submit:
            try:
                with st.spinner('Calculating estimate...'):
                    # 1.  Map from inputs to the CustomData Bridge
                    data_obj = CustomData( 
                        Age=age, 
                        Sex=sex, 
                        BMI=bmi, 
                        Children=children, 
                        Smoker=smoker, 
                        Region=region)

                    # 2. Convert dictionary to DataFrame for the preprocessor 
                    df = data_obj.get_as_df()

                    #3. Generate Prediction
                    prediction = pipeline.predict(df)
            
                    # 4. Display results with financial formatting
                    st.markdown("---")
                    st.metric(
                        label="Estimated Annual Charges", 
                        value=f"${prediction:,.2f}"
                    )
                    st.info("💡 Pro-tip: Factors like smoking status and BMI significantly impact your estimate.")

            except Exception as e:
                st.error(f'An error occured during prediction: {e}')
    
    with tab2:
        st.subheader("Exploratory Data Analysis Report")
        st.write("This report shows the statistical evidence used to build the model.")
        
        # Load the HTML file we converted in Step 1
        with open("src/insurance/notebooks/exploration.html", 'r', encoding='utf-8') as f:
            html_data = f.read()
        
        # Display the notebook as an embedded frame
        components.html(html_data, height=800, scrolling=True)


# Call the function (for testing individually)
if __name__ == "__main__":
    run_insurance_ui()