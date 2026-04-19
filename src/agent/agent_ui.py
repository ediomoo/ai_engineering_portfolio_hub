import streamlit as st
import os
import io
from src.agent.pipeline.agent_logic import MarketAgent

def run_agent_ui():
    # --- UI HEADER ---
    st.header("📈 Agentic Market Analyst")
    st.markdown('This autonomous AI agent researches the **live web** and generates structured reports.')

    # --- USER INPUT ---
    company = st.text_input("Enter Company Name (e.g., NVIDIA, Tesla, Microsoft):").strip()

    # --- AGENT EXECUTION ---
    if st.button("Start Autonomous Research"):
        if company:
            with st.spinner(f'Agent is searching, analyzing and writing a report for {company}...'):
                try:
                    # 1. PRE-CREATE DIRECTORY: Ensure the path exists before the agent runs
                    report_dir = os.path.join("artifacts", "agent")
                    os.makedirs(report_dir, exist_ok=True)
                    
                    # 2. RUN AGENT
                    agent = MarketAgent()
                    report = agent.run_analysis(company)

                    # 3. CAPTURE FILE IMMEDIATELY: Read from disk into Session State (RAM)
                    csv_path = os.path.join(report_dir, "market_report.csv")
                    
                    if os.path.exists(csv_path):
                        with open(csv_path, "rb") as f:
                            # Move data to stable memory
                            st.session_state['csv_bytes'] = f.read()
                    else:
                        # Clean up old data if new file wasn't created
                        if 'csv_bytes' in st.session_state:
                            del st.session_state['csv_bytes']

                    # Store text results
                    st.session_state['last_report'] = report
                    st.session_state['last_company'] = company

                except Exception as e:
                    st.error(f"Agent encountered an error: {e}")
                    st.info("💡 Tip: Check your API keys in Streamlit Secrets.")
        else:
            st.warning("⚠️ Please enter a company name.")

    # --- RESULTS DISPLAY ---
    # We display from session_state so the UI survives the download button click
    if 'last_report' in st.session_state:
        st.divider()
        st.subheader(f"Agent's Final Summary: {st.session_state['last_company']}")
        st.markdown(st.session_state['last_report'])

        # --- DOWNLOAD MANAGEMENT (FROM MEMORY) ---
        if 'csv_bytes' in st.session_state:
            # Format filename safely
            clean_name = st.session_state['last_company'].lower().replace(' ', '_')
            safe_filename = f"{clean_name}_market_report.csv"
            
            st.download_button(
                label="📥 Download Detailed News CSV",
                data=st.session_state['csv_bytes'],
                file_name=safe_filename,
                mime="text/csv",
                key="download_btn" # Unique key to prevent state issues
            )
            st.success("✅ Analysis complete! CSV report is ready.")
        else:
            # If we reach here, the agent likely didn't use the 'save_to_csv' tool successfully
            st.warning('⚠️ The report text was generated, but the detailed CSV file was not created by the agent.')
            st.info("Ensure your agent has a tool named `save_to_csv` and that it actually calls it.")

if __name__ == "__main__":
    run_agent_ui()
