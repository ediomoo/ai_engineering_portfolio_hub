# 🚀 AI Engineering & Machine Learning Portfolio Hub

A unified interactive dashboard showcasing end-to-end AI solutions, ranging from classical machine learning (Regression/Classification) to state-of-the-art Generative AI (RAG) and Autonomous Agents.

**🌐 Live Demo:** [Link to your Streamlit Cloud App]

---

## 📂 Featured Projects

### 1. 🏥 Medical Insurance Cost Predictor (Regression)
*   **Goal:** Estimate annual medical insurance charges based on demographic and health factors.
*   **Tech Stack:** Python, XGBoost, Scikit-Learn, MLflow.
*   **Key Features:** Automated hyperparameter tuning and modular MLOps architecture.

### 2. 💰 Loan Approval System (Classification)
*   **Goal:** Predict loan eligibility based on applicant profiles to streamline financial decision-making.
*   **Tech Stack:** XGBoost, Scikit-Learn.
*   **Key Features:** Clean data transformation pipeline and intuitive user interface.

### 3. 🕵️‍♂️ PDF Investigator (RAG)
*   **Goal:** A "Chat with your PDF" system that retrieves contextually relevant answers from uploaded documents.
*   **Tech Stack:** LangChain, OpenAI Embeddings, FAISS Vector Store.
*   **Key Features:** Recursive character splitting and semantic search retrieval.

### 4. 📈 Agentic Market Analyst (Autonomous Agent)
*   **Goal:** An AI Agent that researches live web news and generates structured market reports autonomously.
*   **Tech Stack:** LangGraph, OpenAI (GPT-4o), Tavily Search API.
*   **Key Features:** Uses the **ReAct** (Reason + Act) framework to browse the web and export findings to CSV.

---

## 🛠️ Tech Stack & Skills
*   **Languages:** Python (3.11+)
*   **AI Frameworks:** LangChain, LangGraph, OpenAI API
*   **Machine Learning:** XGBoost, Scikit-Learn, Pandas, NumPy
*   **DevOps & MLOps:** MLflow, Docker, GitHub Actions, Streamlit Cloud
*   **Vector DB:** FAISS

---

## 🏗️ Project Structure

├── artifacts/          # Saved models, preprocessors, and agent reports
├── data/               # Raw datasets (CSV/PDF)
├── notebooks/          # Jupyter notebooks for EDA and experimentation
├── plots/              # Visualizations and performance charts
├── src/                # Modular source code
│   ├── insurance/      # Insurance project components & pipelines
│   ├── loan/           # Loan project components & pipelines
│   ├── rag/            # RAG components & query pipelines
│   └── agent/          # Agent tools and reasoning logic
├── Dockerfile          # Containerization configuration
├── mainhub.py          # Main Streamlit entry point
├── README.md           # Project documentation
├── setup.py            # Local package configuration
└── requirements.txt    # Project dependencies
	
---

⚙️ Installation & Setup

1. Clone the Repository and change working directory: git clone https://github.com/ediomoo/ai_engineering_portfolio_hub.git
						    : cd ai-engineering-portfolio-hub
2. Environment Configuration
Local Development: Create a .env file in the root directory:
OPENAI_API_KEY=your_openai_api_key
TAVILY_API_KEY=your_tavily_api_key
GITHUB_TOKEN=your_github_token

Streamlit Cloud Deployment:
Add the keys above to the Secrets section in your Streamlit Cloud dashboard settings.

3. Install Dependencies
Install all required libraries and the local project as an editable package: pip install -r requirements.txt

4. Initialize Package (setup.py)
Ensure the local package is recognized by running: pip install -e .

5. Run the Application: streamlit run mainhub.py
