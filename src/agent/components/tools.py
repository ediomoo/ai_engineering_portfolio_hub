import pandas as pd
import os
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool

@tool
def save_to_csv(data_list: list, filename: str = "market_report.csv"):
    """
    Saves a list of dictionaries (e.g., [{'headline': '...', 'sentiment': '...'}]) into a CSV file.
    
    Use this tool ONLY when you have collected structured data points that need 
    to be available for download.
    """
    try:
        # Ensure we are working with a list of data
        if not data_list or not isinstance(data_list, list):
            return "Error: Data must be a non-empty list of dictionaries."

        # Create the directory path for agent artifacts if it doesn't exist
        # 'exist_ok=True' prevents errors if the folder already exists
        target_dir = os.path.join("artifacts", "agent")
        os.makedirs(target_dir, exist_ok=True)

        # Construct the full file path
        file_path = os.path.join(target_dir, filename)

        # Use pandas to handle potential formatting issues automatically
        df = pd.DataFrame(data_list)
        df.to_csv(file_path, index=False)

        # Return a clear success message so the agent knows the task is done
        return f"Successfully saved {len(data_list)} rows to {file_path}"
    
    except Exception as e:
        return f"Failed to save CSV: {str(e)}"

def get_tools():
    """
    Initializes and returns the suite of tools available to the agent.
    """
    # Initialize Tavily search with a limit of 5 results for concise context
    # Note: Ensure TAVILY_API_KEY is in your environment variables
    search_tool = TavilySearchResults(k=5)
    
    # Corrected: Fixed the indentation of the return statement
    return [search_tool, save_to_csv]
