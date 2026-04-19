from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from src.agent.components.tools import get_tools
import os

class MarketAgent:
    """
    Orchestrates the autonomous market analysis process using a ReAct agent.
    Ensures that news data is physically saved to CSV before summarizing.
    """
    def __init__(self):
        # Initialize LLM - Using temperature 0 for factual financial precision
        self.llm = ChatOpenAI(
            base_url="https://models.inference.ai.azure.com",
            api_key=os.environ.get("GITHUB_TOKEN"),
            model="gpt-4o",
            temperature=0
        )
        
        self.tools = get_tools()
        
        # MANDATORY SYSTEM INSTRUCTIONS:
        # We explicitly tell the agent that saving the CSV is a requirement for success.
        system_message = (
    "You are a professional Financial Analyst. Your workflow is strict:\n"
    "1. Search for news from the last 7 days.\n"
    "2. Extract headlines and assign a sentiment (Positive/Negative/Neutral).\n"
    "3. You MUST call the 'save_to_csv' tool with the structured list.\n"
    "4. Provide a final summary and Buy/Hold/Sell recommendation.\n"
    "IMPORTANT: Do NOT include any 'Download here' links in your text. "
    "The system will provide a download button automatically."
)


        # Create the ReAct agent
        # Note: In LangGraph 0.3+, 'prompt' is the correct argument for system instructions
        self.agent_executor = create_react_agent(
            self.llm, 
            self.tools, 
            prompt=system_message
        )

    def run_analysis(self, company):
        """
        Executes the research loop for a specific company.
        """
        # User prompt that reinforces the tool-use requirement
        user_input = (
            f"Perform a detailed market analysis for {company}.\n"
            f"- Gather news from the last 7 days.\n"
            "- Save the structured results (Headline, Sentiment) to CSV using your tool.\n"
            "- Summarize the findings and give a Buy/Hold/Sell recommendation."
        )

        inputs = {"messages": [{"role": "user", "content": user_input}]}
        
        # Invoke the agent
        result = self.agent_executor.invoke(inputs)

        # Return the final message content
        return result["messages"][-1].content
