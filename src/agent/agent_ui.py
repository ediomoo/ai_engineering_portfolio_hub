import streamlit as st
import os
import io
from src.agent.pipeline.agent_logic import MarketAgent

def run_agent_ui():
    st.header("📈 Agentic Market Analyst")
    st.markdown('This autonomous AI agent researches the **live web** and generates structured reports.')

    company = st.text_input("Enter Company Name (e.g., NVIDIA, Tesla, Microsoft):").strip()

    if st.button("Start Autonomous Research"):
        if company:
            with st.spinner(f'Agent is searching...'):
                try:
                    agent = MarketAgent()
                    report = agent.run_analysis(company)

                    # --- NEW LOGIC: Capture file immediately ---
                    csv_path = os.path.join("artifacts", "agent", "market_report.csv")
                    
                    if os.path.exists(csv_path):
                        with open(csv_path, "rb") as f:
                            # Store the actual BYTES in session state
                            st.session_state['csv_bytes'] = f.read()
                    
                    st.session_state['last_report'] = report
                    st.session_state['last_company'] = company

                except Exception as e:
                    st.error(f"Agent encountered an error: {e}")
        else:
            st.warning("⚠️ Please enter a company name.")

    # --- RESULTS DISPLAY ---
    if 'last_report' in st.session_state:
        st.subheader(f"Agent's Final Summary: {st.session_state['last_company']}")
        st.markdown(st.session_state['last_report'])

        # --- DOWNLOAD FROM MEMORY (Not Disk) ---
        if 'csv_bytes' in st.session_state:
            safe_filename = f"{st.session_state['last_company'].lower().replace(' ', '_')}_report.csv"
            
            st.download_button(
                label="📥 Download Detailed News CSV",
                data=st.session_state['csv_bytes'], # Use the stored bytes
                file_name=safe_filename,
                mime="text/csv"
            )
            st.success("Analysis complete! CSV report is ready.")
        else:
            st.warning('The report was generated, but the CSV data could not be retrieved.')

if __name__ == "__main__":
    run_agent_ui()
