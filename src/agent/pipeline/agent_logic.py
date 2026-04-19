from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from src.agent.components.tools import get_tools
import os

class MarketAgent:
    """
    Orchestrates the autonomous market analysis process using a ReAct agent.
    The agent uses real-time search and data tools to provide investment insights.
    """
    def __init__(self):
        """
        Initializes the LLM and binds the custom toolset to the agent executor.
        Uses GPT-4o for high-reasoning capabilities required for sentiment analysis.
        """

        # Temperature is set to 0 to ensure consistent, analytical reasoning
        self.llm = ChatOpenAI(base_url="https://models.inference.ai.azure.com", # GitHub's endpoint
            api_key=os.environ["GITHUB_TOKEN"],               # Use your GitHub Token
            model="gpt-4o",                                   # Or 'gpt-4o-mini', 'Phi-3-small-8k-instruct'
            temperature=0)
        
        self.tools = get_tools()
        
        # System Message to define the agent's behavior
        # This keeps the free model focused and efficient
        system_message = (
            "You are a professional Financial Analyst. "
            "Gather 7 days of news, analyze sentiment, save to CSV, and provide a clear recommendation."
        )

        # Create the ReAct agent which can autonomously decide which tool to use
        self.agent_executor = create_react_agent(self.llm, self.tools, prompt = system_message)

    def run_analysis(self, company):
        """
        Executes a multi-step financial research workflow for a specific company.
        
        The workflow involves:
        1. Web searching for recent news.
        2. Analyzing headline sentiment.
        3. Persisting findings to CSV.
        4. Synthesizing a final investment recommendation.
        """
       
        # Defining a structured multi-step instruction for the agent's reasoning loop
        prompt = (
            f"1. Search for the latest news on {company} for the last 7 days.\n"
            f"2. Analyze the sentiment (Positive, Negative, or Neutral) for each headline.\n"
            f"3. Use the 'save_to_csv' tool to store a list of headlines and sentiments.\n"
            f"4. Provide a final summary: Based on this data, is it a good time to buy, sell or hold?"
        )

        # Preparing the input in the standard chat message format
        inputs = {"messages": [{"role": "user", "content": prompt}]}
        
        # The agent invokes the reasoning loop and tool-calling sequence
        result = self.agent_executor.invoke(inputs)

        # Returning the final content from the last message in the agent's response chain
        return result["messages"][-1].content