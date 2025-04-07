import streamlit as st
import requests
import json
import os
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from langchain_community.tools import DuckDuckGoSearchRun
import time 
import tiktoken
from collections import deque


# Globals for token tracking
TPM_LIMIT = 6000
token_window = deque()  # stores (timestamp, tokens_used)
WINDOW_SECONDS = 60  # seconds
window_duration=60
# Load environment variables
load_dotenv()

# Get Grok API key from environment variables
GROK_API_KEY = os.getenv("GROK_API_KEY")

# Page configuration
st.set_page_config(page_title="Fake News Detector", page_icon="in", layout="wide")

# Add a sidebar with information
st.sidebar.title("About")
st.sidebar.info(
    "This app uses multiple AI agents powered by Grok API to analyze news articles "
    "and predict if they contain false or misleading information. Enter a news article "
    "in the main panel to get started."
)

# Add a footer
st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #0E1117;
        color: #FAFAFA;
        text-align: center;
        padding: 10px;
        font-size: 14px;
    }
    </style>
    <div class="footer">
        Developed by Meghana,Bhuvan and Danush
    </div>
    """,
    unsafe_allow_html=True
)

# Set up the Streamlit app
st.title("Fake News Detector")

# Temperature slider
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.1, step=0.1)

# Input for news article
news_input = st.text_area("Enter the news article to analyze:")

# Initialize the search tool
search_tool = DuckDuckGoSearchRun()

# Estimate token count
def count_tokens(text):
    enc = tiktoken.get_encoding("cl100k_base")  # Approximate encoder
    return len(enc.encode(text))

class GrokAgent:
    def __init__(self, role, goal, backstory, temperature=0.1):
        self.role = role
        self.goal = goal
        self.backstory = backstory
        self.temperature = temperature
        self.search_tool = DuckDuckGoSearchRun()
    def wait_for_token_budget(self, tokens_needed):
        while True:
            now = time.time()

            # Remove tokens outside the 60-second window
            while token_window and now - token_window[0][0] > WINDOW_SECONDS:
                token_window.popleft()

            used_tokens = sum(t[1] for t in token_window)

            if used_tokens + tokens_needed <= TPM_LIMIT:
                # Safe to proceed
                token_window.append((now, tokens_needed))
                return
            else:
                # Wait until enough tokens are free
                earliest_time, earliest_tokens = token_window[0]
                wait_time = WINDOW_SECONDS - (now - earliest_time) + 0.1
                print(f"[TPM] Limit hit. Waiting for {wait_time:.2f} seconds...")
                time.sleep(wait_time)
    
    def wait_for_tpm(self, tokens_needed):
        now = time.time()
        # Remove old entries from the window
        while token_window and (now - token_window[0][0]) > window_duration:
            token_window.popleft()

        current_tpm = sum(tokens for t, tokens in token_window)

        if current_tpm + tokens_needed > TPM_LIMIT:
            wait_time = window_duration - (now - token_window[0][0])
            print(f"[TPM limit] Sleeping for {wait_time:.2f} seconds...")
            time.sleep(wait_time)
            self.wait_for_tpm(tokens_needed)  # recursive retry

    def call_grok_api(self, prompt):
        """Make a call to the Grok API"""

        tokens_needed = count_tokens(prompt)

        # Wait for permission based on TPM
        self.wait_for_token_budget(tokens_needed)

        headers = {
            "Authorization": f"Bearer {GROK_API_KEY}",
            "Content-Type": "application/json"
        }

        data = {
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "model": "deepseek-r1-distill-qwen-32b"  # Using Grok-1 model
        }
        max_retries=3

        for attempt in range(max_retries):
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json=data
            )

            if response.status_code == 200:
                content = response.json()["choices"][0]["message"]["content"]
                # Add output tokens as well to the window
                output_tokens = count_tokens(content)
                self.wait_for_token_budget(output_tokens)
                return content

            elif response.status_code == 429:
                print(f"[Retry {attempt+1}] 429 Too Many Requests. Backing off...")
                time.sleep(10)
            else:
                raise Exception(f"API Error {response.status_code}: {response.text}")

        raise Exception("Failed after max retries due to rate limiting.")
    def execute_task(self, task_description, context, expected_output):
        """Execute a task based on the agent's role and goal"""
        try:
            # Perform search if needed
            search_results = self.search_tool.run(f"{self.role} researching: {context[:200]}")
        except:
            search_results = "Unable to retrieve search results."
        
        prompt = f"""
        # Task Assignment for {self.role}
        
        ## Your Role
        {self.role}
        
        ## Your Goal
        {self.goal}
        
        ## Your Backstory
        {self.backstory}
        
        ## Task Description
        {task_description}
        
        ## Context
        {context}
        
        ## External Information
        {search_results}
        
        ## Expected Output Format
        {expected_output}
        
        Complete your assigned task based on your role and goal. Be thorough and accurate.
        """
        
        response = self.call_grok_api(prompt)
        return response

def create_agents(temp):
    """Create the specialized agents"""
    researcher = GrokAgent(
        role="Research Analyst",
        goal="Find relevant information about the news article and identify its main claims",
        backstory="You are an expert in political news and current events with a keen eye for identifying key claims",
        temperature=temp
    )
    
    fact_checker = GrokAgent(
        role="Fact Checker",
        goal="Verify the claims made in the news article using reliable sources",
        backstory="You are a meticulous fact-checker with years of experience in journalism and fact verification",
        temperature=temp
    )
    
    bias_detector = GrokAgent(
        role="Bias Detector",
        goal="Identify political bias, emotional language, and framing techniques in the article",
        backstory="You are an expert in media literacy who specializes in detecting bias in news reporting",
        temperature=temp
    )
    
    authenticator = GrokAgent(
        role="Source Authenticator",
        goal="Evaluate the credibility of the article's sources and author",
        backstory="You are a specialist in source verification with expertise in identifying credible vs. questionable information sources",
        temperature=temp
    )
    
    return [researcher, fact_checker, bias_detector, authenticator]

def create_tasks(news_article):
    """Create tasks for the agents"""
    return [
        {
            "description": "Identify and analyze the main claims and entities in the news article. Extract the key assertions being made.",
            "agent_index": 0,  # researcher
            "expected_output": "A detailed list of the main claims in the article, the entities involved, and a brief background on each claim's context."
        },
        {
            "description": "Fact-check the main claims identified by the researcher using reliable sources. Verify if each claim is true, false, misleading, or unverifiable.",
            "agent_index": 1,  # fact_checker
            "expected_output": "A systematic evaluation of each claim with verification status, evidence, and reliable sources."
        },
        {
            "description": "Analyze the article for bias indicators including partisan language, emotional manipulation, framing, and narrative techniques.",
            "agent_index": 2,  # bias_detector
            "expected_output": "A report on detected bias with specific examples, political leaning assessment, and emotionality analysis."
        },
        {
            "description": "Evaluate the credibility of the sources cited in the article, the author's credentials, and the publishing platform's reputation.",
            "agent_index": 3,  # authenticator
            "expected_output": "A credibility assessment of the article's sources, author background check, and publishing platform evaluation."
        }
    ]

def run_agent_task(agent, task, news_article):
    """Run a single agent task"""
    return {
        "agent_role": agent.role,
        "task": task["description"],
        "result": agent.execute_task(task["description"], news_article, task["expected_output"])
    }

def integrate_results(agent_results):
    """Integrate results from all agents into a final report"""
    integrator = GrokAgent(
        role="Lead Analyst",
        goal="Integrate all findings into a cohesive assessment of the article's reliability",
        backstory="You are the lead analyst who synthesizes reports from multiple experts to deliver comprehensive fake news assessments",
        temperature=0.1
    )
    
    integrator_prompt = f"""
    # Final Report Integration
    
    As the Lead Analyst, synthesize these specialist reports into a comprehensive assessment:
    
    ## Research Analyst's Findings:
    {agent_results[0]["result"]}
    
    ## Fact Checker's Findings:
    {agent_results[1]["result"]}
    
    ## Bias Detector's Findings:
    {agent_results[2]["result"]}
    
    ## Source Authenticator's Findings:
    {agent_results[3]["result"]}
    
    ## Instructions:
    Create a comprehensive, integrated report that synthesizes these findings.
    Include:
    1. Executive Summary with overall credibility score (0-100%)
    2. Key Claims Assessment
    3. Fact-Check Results
    4. Bias Analysis
    5. Source Credibility
    6. Final Verdict on reliability
    7. Recommendations for readers
    
    Format your response as a well-structured report with clear sections and a summary dashboard at the top.
    """
    
    return integrator.call_grok_api(integrator_prompt)

if st.button("Analyze News", key="analyze_news_button"):
    if news_input:
        if not GROK_API_KEY:
            st.error("Grok API key not found. Please set the GROK_API_KEY environment variable.")
        else:
            with st.spinner("Multiple Grok agents are analyzing the article... This may take a moment."):
                try:
                    # Create agents
                    agents = create_agents(temperature)
                    
                    # Create tasks
                    tasks = create_tasks(news_input)
                    
                    # Progress indicators
                    progress_placeholder = st.empty()
                    progress_bar = progress_placeholder.progress(0)
                    status_placeholder = st.empty()
                    
                    # Execute tasks sequentially (can be made parallel if needed)
                    agent_results = []
                    
                    for i, task in enumerate(tasks):
                        status_placeholder.text(f"Agent {i+1}/{len(tasks)}: {agents[task['agent_index']].role} is working...")
                        progress_bar.progress((i) / len(tasks))
                        
                        # Run agent task
                        result = run_agent_task(agents[task["agent_index"]], task, news_input)
                        agent_results.append(result)
                    
                    progress_bar.progress((len(tasks)-1)/len(tasks))
                    status_placeholder.text("Integrating results from all agents...")
                    
                    # Integrate results
                    final_report = integrate_results(agent_results)
                    
                    # Clear progress indicators
                    progress_placeholder.empty()
                    status_placeholder.empty()
                    
                    # Create tabs for individual reports and final report
                    tab_final, tab_research, tab_fact, tab_bias, tab_source = st.tabs([
                        "Final Analysis", 
                        "Research Report", 
                        "Fact Check", 
                        "Bias Analysis",
                        "Source Credibility"
                    ])
                    
                    with tab_final:
                        st.markdown("## Integrated Analysis Report")
                        st.markdown(final_report)
                    
                    with tab_research:
                        st.markdown(f"## {agent_results[0]['agent_role']} Report")
                        st.markdown(agent_results[0]["result"])
                    
                    with tab_fact:
                        st.markdown(f"## {agent_results[1]['agent_role']} Report")
                        st.markdown(agent_results[1]["result"])
                    
                    with tab_bias:
                        st.markdown(f"## {agent_results[2]['agent_role']} Report")
                        st.markdown(agent_results[2]["result"])
                    
                    with tab_source:
                        st.markdown(f"## {agent_results[3]['agent_role']} Report")
                        st.markdown(agent_results[3]["result"])
                    
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
    else:
        st.warning("Please enter a news article to analyze.")

# Add API key instructions in the sidebar
st.sidebar.markdown("## API Key Setup")
st.sidebar.info(
    "This application requires a Grok API key. "
    "Set your GROK_API_KEY as an environment variable or in a .env file."
)