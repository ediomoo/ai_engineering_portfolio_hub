import streamlit as st
import os
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
                    # Initialize and run the agent
                    agent = MarketAgent()
                    report = agent.run_analysis(company)

                    # Store report in session state so it persists during downloads
                    st.session_state['last_report'] = report
                    st.session_state['last_company'] = company

                except Exception as e:
                    st.error(f"Agent encountered an error: {e}")
                    st.info("💡 Make sure your TAVILY_API_KEY and OPENAI_API_KEY are set in Streamlit Secrets.")
        else:
            st.warning("⚠️ Please enter a company name.")

    # --- RESULTS DISPLAY ---
    # We display from session_state so the UI doesn't clear when clicking "Download"
    if 'last_report' in st.session_state:
        st.subheader(f"Agent's Final Summary: {st.session_state['last_company']}")
        st.markdown(st.session_state['last_report'])

        # --- DOWNLOAD MANAGEMENT ---
        csv_path = os.path.join("artifacts", "agent", "market_report.csv")

        if os.path.exists(csv_path):
            with open(csv_path, "rb") as file:
                # Corrected: Fixed nested double quotes in the f-string (use single quotes inside)
                safe_filename = f"{st.session_state['last_company'].lower().replace(' ', '_')}_market_report.csv"
                
                st.download_button(
                    label="📥 Download Detailed News CSV",
                    data=file,
                    file_name=safe_filename,
                    mime="text/csv"
                )
            st.success("Analysis complete! CSV report is ready for download.")
        else:
            st.warning('The report was generated, but no detailed CSV was found in artifacts.')

if __name__ == "__main__":
    run_agent_ui()
