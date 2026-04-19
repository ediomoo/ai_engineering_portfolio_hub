import pandas as pd
import os
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool

@tool
def save_to_csv(data_list: list):
    """
    Saves a list of dictionaries into a CSV file. 
    Example input: [{'Date': '2024-01-01', 'Headline': '...', 'Sentiment': 'Positive'}]
    
    Use this tool IMMEDIATELY after gathering news and analyzing sentiment.
    The file will be stored at 'artifacts/agent/market_report.csv'.
    """
    try:
        # 1. Validation: Ensure we have a list of dictionaries
        if not data_list or not isinstance(data_list, list):
            return "Error: Data must be a list of dictionaries."

        # 2. Path Setup: Hardcode the path to ensure UI and Agent stay synced
        target_dir = os.path.join("artifacts", "agent")
        os.makedirs(target_dir, exist_ok=True)
        file_path = os.path.join(target_dir, "market_report.csv")

        # 3. Save Logic
        df = pd.DataFrame(data_list)
        df.to_csv(file_path, index=False)

        # 4. Confirmation: Clear feedback to the Agent
        return f"SUCCESS: Saved {len(data_list)} news items to {file_path}. Tell the user the CSV is ready for download."
    
    except Exception as e:
        return f"CRITICAL ERROR: Failed to save CSV: {str(e)}"

def get_tools():
    """
    Initializes and returns the tools.
    """
    # Increased k to 7 to give the agent more data to analyze sentiment
    search_tool = TavilySearchResults(k=7)
    
    return [search_tool, save_to_csv]
