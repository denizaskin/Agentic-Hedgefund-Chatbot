import os
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"

###############################################################################
# IMPORTS
###############################################################################
import streamlit as st
import torch
import re
import json
import PyPDF2
import threading
import queue
import time
from dotenv import load_dotenv
from typing_extensions import TypedDict
from typing import List
import altair as alt
import pandas as pd
import numpy as np
import gym
from gym import spaces
import yfinance as yf
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import matplotlib.pyplot as plt
import sys
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform
from sklearn.covariance import LedoitWolf
from pydantic import BaseModel, Field, ValidationError

###############################################################################
# PERFORMANCE / SETUP
###############################################################################
torch.set_num_threads(os.cpu_count())
torch.set_flush_denormal(True)
st.set_page_config(
    page_title="Agentic Hedge Fund Chatbot",
    layout="wide"
)

load_dotenv()
url = os.getenv("WATSONX_URL")
apikey = os.getenv("WATSONX_APIKEY")
project_id = os.getenv("WATSONX_PROJECT_ID")
openai_apikey = os.getenv("OPENAI_API_KEY")  # optional

###############################################################################
# LLMs
###############################################################################
from langchain_ibm import ChatWatsonx
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate

model_id_llama = "meta-llama/llama-3-405b-instruct"
parameters = {
    "decoding_method": "greedy",
    "max_new_tokens": 10000,
    "min_new_tokens": 1,
    "temperature": 0,
    "top_k": 1,
    "top_p": 1.0,
    "seed": 42
}

llm_llama = ChatWatsonx(
    model_id=model_id_llama,
    url=url,
    apikey=apikey,
    project_id=project_id,
    params=parameters
)

llm_chat_gpt = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key=openai_apikey
)

# We'll use GPT-4o by default:
llm = llm_chat_gpt

###############################################################################
# HELPER: CONTROL-CHAR SANITIZATION
###############################################################################
def sanitize_control_chars(text: str) -> str:
    import re
    return re.sub(r'[\x00-\x1f\x7f]+', ' ', text)

###############################################################################
# PDF READER
###############################################################################
@st.cache_data
def read_pdf_text(file_path: str) -> str:
    text_content = []
    with open(file_path, "rb") as pdf_file:
        reader = PyPDF2.PdfReader(pdf_file)
        for page in reader.pages:
            text_content.append(page.extract_text())
    return "\n".join(text_content)

###############################################################################
# SUBTASK REWRITER FOR USER => question
###############################################################################
def rewrite_subtask_as_question(subtask_text: str) -> str:
    """
    Use the LLM to transform a USER subtask into a direct question.
    """
    prompt = f"""
You are an AI that transforms instructions into a question.
The original subtask text is:
'{subtask_text}'

Please rewrite it as a concise question the user can answer in one or two lines.
Make it polite and direct.
"""
    from langchain.schema import HumanMessage
    try:
        response_msg = llm.invoke([HumanMessage(content=prompt)])
        raw_text = response_msg.content if hasattr(response_msg, "content") else response_msg[0].content
        return sanitize_control_chars(raw_text).strip()
    except Exception as e:
        return f"Error rewriting subtask as question: {e}"

###############################################################################
# CHATBOT WORKFLOW + PROMPTS
###############################################################################
from langgraph.graph import END, StateGraph

PLANNER_PROMPT_TEXT = """
You are a question answering agent whose task is to look at a PDF document which
is an official Investment Statement for the Base Canada Pension Plan (bCPP) and 
the Additional Canada Pension Plan (aCPP) investment portfolios. It details the 
key investment objectives, policies, return expectations, risk management 
strategies, asset allocation methods, valuation techniques, and sustainable 
investing practices that guide how CPP Investments manages these 
long-horizon portfolios. Essentially, it sets the framework for ensuring the CPP 
remains self-sustaining while balancing risk and return in line with legislative 
mandates and stakeholder interests., and your task is to create a sequence of 
subtasks needed to address the user’s question thoroughly.

Context Provided:
1) PDF Content: {{pdf_text}}
2) Existing user question/input: "{{user_question}}"

Your Objective:
1) Figure out each step or piece of information required to provide a complete, 
   accurate answer.
2) If the needed information is already available in the context (the PDF text 
   or the user’s previous answers), you can answer with an "LLM" subtask.
3) If any piece of information is missing—i.e., it is not in the PDF text or 
   the user’s previous answers—then you must not guess or assume. Instead, 
   you must create a "USER" subtask to explicitly ask the user for that 
   missing detail.

Instructions:
- Return only valid JSON, structured as follows:
{
  "subtasks": [
    {
      "task": "Subtask text here",
      "type": "LLM or USER"
    },
    ...
  ]
}
- "type": "LLM" if the subtask is fully solvable with the provided PDF text + 
  any known user inputs.
- "type": "USER" if you do NOT have all necessary data to complete that subtask 
  using only the PDF text and prior user inputs. In that case, prompt the user 
  for the missing details.

Remember:
- Do NOT fabricate user data or guess any missing numerical values, dates, or 
  other specifics.
- Do NOT provide your final reasoning or steps outside of JSON.
- Only produce valid JSON in the exact format shown above, with no extra commentary.
"""

from pydantic import BaseModel, Field

class AgentWorkflowState(TypedDict):
    pdf_path: str
    user_question: str
    pdf_text: str
    planner_output: str
    execution_result: dict
    final_answer: str

class SubTask(BaseModel):
    task: str = Field(..., description="Subtask description")
    type: str = Field(..., description="'LLM' or 'USER'")

class PlannerOutput(BaseModel):
    subtasks: List[SubTask] = Field(..., description="List of required subtasks")

def stream_llm_call(prompt: str) -> str:
    from langchain.schema import HumanMessage
    try:
        response_msg = llm.invoke([HumanMessage(content=prompt)])
        raw_text = response_msg.content if hasattr(response_msg, "content") else response_msg[0].content
        return sanitize_control_chars(raw_text)
    except Exception as e:
        return f"Error: LLM call failed ({e})"

def compliance_planner_node(state: AgentWorkflowState) -> AgentWorkflowState:
    pdf_text = state["pdf_text"]
    user_question = state["user_question"]
    final_planner_prompt = PLANNER_PROMPT_TEXT.replace("{{pdf_text}}", pdf_text)\
                                             .replace("{{user_question}}", user_question)
    raw_output = stream_llm_call(final_planner_prompt)
    from langchain.output_parsers import PydanticOutputParser
    parser = PydanticOutputParser(pydantic_object=PlannerOutput)
    try:
        structured_output = parser.parse(raw_output)
        planner_response = structured_output.model_dump_json()
    except Exception as e:
        planner_response = f'{{"subtasks":[],"error":"{e}"}}'
    state["planner_output"] = planner_response
    return state

def executer_node(state: AgentWorkflowState) -> AgentWorkflowState:
    return state

def question_answerer_node(state: AgentWorkflowState) -> AgentWorkflowState:
    pdf_text = state["pdf_text"]
    user_question = state["user_question"]
    planner_output = state["planner_output"]
    execution_result = state["execution_result"]
    subtask_answers = execution_result.get("subtask_answers", [])
    context_str = ""
    for i, ans in enumerate(subtask_answers):
        context_str += f"[Subtask #{i+1} - {ans['type']}]\nTask: {ans['task']}\n\nAnswer: {ans['answer']}\n\n"

    final_prompt = f"""
You are the Question Answerer Agent.

You have:
1) PDF Content:
{pdf_text}

2) The user's question:
{user_question}

3) The plan (raw JSON):
{planner_output}

4) The completed subtasks and their answers:
{context_str}

Provide a final consolidated answer that addresses the user's question in full detail.
"""
    final_answer = stream_llm_call(final_prompt).strip()
    state["final_answer"] = final_answer
    return state

from langgraph.graph import END, StateGraph

def build_workflow():
    wf = StateGraph(AgentWorkflowState)
    wf.set_entry_point("compliance_planner")
    wf.add_node("compliance_planner", compliance_planner_node)
    wf.add_node("executer_agent", executer_node)
    wf.add_node("question_answerer", question_answerer_node)
    wf.add_edge("compliance_planner", "executer_agent")
    wf.add_edge("executer_agent", "question_answerer")
    wf.add_edge("question_answerer", END)
    return wf

###############################################################################
# HRP PARAMS MODEL & DECIDER
###############################################################################
class HRPParams(BaseModel):
    tickers: List[str] = Field(default=["SPY","EFA","EEM","AGG","LQD","GLD","DBC"])
    start_date: str = "2010-01-01"
    end_date: str = "2023-01-01"
    max_weight: float = None
    method: str = "ward"

def hrp_param_decider(chatbot_answer: str) -> dict:
    from langchain.output_parsers import PydanticOutputParser
    from pydantic import ValidationError

    parser = PydanticOutputParser(pydantic_object=HRPParams)
    prompt = f"""
You are an HRP parameter decider.
Given the chatbot's final answer:

{chatbot_answer}

Please return valid JSON containing HRP parameters. Must match this schema:
{{
  "tickers": ["string", ...],
  "start_date": "YYYY-MM-DD",
  "end_date": "YYYY-MM-DD",
  "max_weight": 0.3,
  "method": "ward"/"complete"/"single"
}}
No extra keys. No explanation. Just valid JSON.
"""
    raw_text = stream_llm_call(prompt)
    try:
        structured = parser.parse(raw_text)
        return structured.model_dump()
    except ValidationError as e:
        print(f"Validation error: {e}")
        return {}

###############################################################################
# HYPERPARAM DECIDER
###############################################################################
def hyperparameter_decider(hrp_output: str) -> str:
    """
    This function instructs the LLM to decide TD3 hyperparameters (actor_lr, critic_lr, gamma, tau, etc.)
    from the HRP output.
    """
    prompt = f"""
You are a hyperparameter decider for a TD3 trading agent.
Given this HRP output:
{hrp_output}

Return valid JSON with keys like "actor_lr", "critic_lr", "gamma", "tau", "trading_style" etc.
No extra text or commentary, just valid JSON.
"""
    return stream_llm_call(prompt).strip()

###############################################################################
# GET PREPROCESSED DATA => df_merged
###############################################################################
def get_preprocessed_data():
    spy_raw = yf.download("SPY", start="2010-01-01", end="2020-01-01", auto_adjust=True)
    qqq_raw = yf.download("QQQ", start="2010-01-01", end="2020-01-01", auto_adjust=True)

    def compute_SMA(series, window=14):
        return series.rolling(window=window).mean()
    def compute_RSI(series, window=14):
        delta = series.diff()
        gain = delta.clip(lower=0).rolling(window=window).mean()
        loss = -delta.clip(upper=0).rolling(window=window).mean()
        rs = gain / (loss + 1e-6)
        return 100 - (100 / (1 + rs))
    def compute_bollinger_bands(series, window=20, num_std=2):
        sma = series.rolling(window=window).mean()
        std = series.rolling(window=window).std()
        upper = sma + num_std * std
        lower = sma - num_std * std
        return sma, upper, lower
    def compute_momentum(series, window=5):
        return series.pct_change(window) * 100
    def compute_volatility(series, window=20):
        returns = series.pct_change()
        return returns.rolling(window=window).std()
    def compute_vix(series, window=20):
        vol = compute_volatility(series, window=window)
        return vol * np.sqrt(252) * 100
    def add_technical_indicators(df):
        df['SMA'] = compute_SMA(df['Close'])
        df['RSI'] = compute_RSI(df['Close'])
        sma, bb_upper, bb_lower = compute_bollinger_bands(df['Close'])
        df['BB_upper'] = bb_upper
        df['BB_lower'] = bb_lower
        df['Momentum'] = compute_momentum(df['Close'])
        df['Volatility'] = compute_volatility(df['Close'])
        return df.dropna()

    spy_df = add_technical_indicators(spy_raw[['Close']].copy())
    qqq_df = add_technical_indicators(qqq_raw[['Close']].copy())

    spy_df = spy_df.rename(columns={
        'Close': 'Close_spy_SPY',
        'SMA': 'SMA_spy_',
        'RSI': 'RSI_spy_',
        'BB_upper': 'BB_upper_spy_',
        'BB_lower': 'BB_lower_spy_',
        'Momentum': 'Momentum_spy_',
        'Volatility': 'Volatility_spy_'
    })
    qqq_df = qqq_df.rename(columns={
        'Close': 'Close_qqq_QQQ',
        'SMA': 'SMA_qqq_',
        'RSI': 'RSI_qqq_',
        'BB_upper': 'BB_upper_qqq_',
        'BB_lower': 'BB_lower_qqq_',
        'Momentum': 'Momentum_qqq_',
        'Volatility': 'Volatility_qqq_'
    })
    spy_raw['VIX_proxy'] = compute_vix(spy_raw['Close'])
    np.random.seed(42)
    spy_raw['Sentiment'] = np.random.uniform(-1, 1, size=len(spy_raw))
    spy_vix_sent = spy_raw[['VIX_proxy', 'Sentiment']].dropna()

    spy_df = spy_df.loc[spy_vix_sent.index]
    qqq_df = qqq_df.loc[spy_vix_sent.index]

    merged_df = pd.concat([spy_df.reset_index(drop=True),
                           qqq_df.reset_index(drop=True),
                           spy_vix_sent.reset_index(drop=True)], axis=1)
    merged_df.dropna(inplace=True)
    merged_df.reset_index(drop=True, inplace=True)
    return merged_df

df_merged = get_preprocessed_data()

###############################################################################
# ADVANCED MULTI ASSET ENV
###############################################################################
class AdvancedMultiAssetTradingEnv(gym.Env):
    """
    Base environment for multi-asset trading. We'll rely on a child class
    or step override to do actual reward logic.
    """
    def __init__(self, df, initial_balance=100000, commission=0.0005, slippage_pct=0.0005, spread=0.001):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.initial_balance = float(initial_balance)
        self.commission = commission
        self.slippage_pct = slippage_pct
        self.spread = spread
        self.action_space = spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(19,), dtype=np.float32)
        self.reset()

    def reset(self):
        self.balance = float(self.initial_balance)
        self.portfolio_value = float(self.initial_balance)
        self.peak_value = float(self.initial_balance)
        self.alloc_spy = 0.0
        self.alloc_qqq = 0.0
        self.current_step = 0
        self.max_drawdown = 0.0
        return self._get_state()

    def _get_state(self):
        if self.current_step >= len(self.df):
            self.current_step = len(self.df) - 1
        # We'll produce a dummy 19-dim state
        # A child class can override step() to produce real data
        return np.zeros((19,), dtype=np.float32)

    def step(self, action):
        self.current_step += 1
        done = (self.current_step >= len(self.df) - 1)
        reward = float(0.0)
        next_state = self._get_state()
        return next_state, reward, done, {}

    def render(self):
        print(f"Step: {self.current_step}, Value: {self.portfolio_value:.2f}, "
              f"SPY: {self.alloc_spy:.2f}, QQQ: {self.alloc_qqq:.2f}")

###############################################################################
# HRP RUN
###############################################################################
def run_hrp_inline(chatbot_answer: str, hrp_params: dict) -> str:
    import io
    old_stdout = sys.stdout
    log_buffer = io.StringIO()

    import yfinance as yf
    from sklearn.covariance import LedoitWolf
    import scipy.cluster.hierarchy as sch
    from scipy.spatial.distance import squareform

    sys.stdout = log_buffer
    try:
        print("----- HRP Agent Informed by Chatbot Output -----\n")
        print(f"Chatbot final answer says: {chatbot_answer}\n")

        tickers_list = hrp_params.get("tickers", ["SPY","EFA","EEM","AGG","LQD","GLD","DBC"])
        if not tickers_list or len(tickers_list) == 0:
            tickers_list = ["SPY","EFA","EEM","AGG","LQD","GLD","DBC"]

        start_date = hrp_params.get("start_date", "2010-01-01")
        end_date = hrp_params.get("end_date", "2023-01-01")
        method = hrp_params.get("method", "ward")
        max_w = hrp_params.get("max_weight", None)

        print(f"HRP params (from chatbot-decided JSON): {hrp_params}\n")
        print(f"Using tickers={tickers_list}, date range=[{start_date}..{end_date}], method={method}\n")

        print("Attempting to download data from yfinance for these tickers...")
        data = yf.download(tickers_list, start=start_date, end=end_date, auto_adjust=True)
        if data is None or data.empty:
            print("No data found with user-chosen dates. Trying fallback: 2010-01-01..2023-01-01")
            data = yf.download(tickers_list, start="2010-01-01", end="2023-01-01", auto_adjust=True)
            if data is None or data.empty:
                raise ValueError("No data returned from yfinance after fallback either!")
        data = data.ffill().dropna()
        data = data['Close']
        print("Downloaded data from yfinance successfully.\n")

        daily_returns = data.pct_change().fillna(0)

        def compute_robust_covariance(returns):
            lw = LedoitWolf().fit(returns)
            return pd.DataFrame(lw.covariance_, index=returns.columns, columns=returns.columns)

        def compute_distance_matrix(corr):
            return np.sqrt(0.5 * (1 - corr))

        def get_quasi_diag(linkage):
            n = linkage.shape[0] + 1
            sort_ix = [int(linkage[-1, 0]), int(linkage[-1, 1])]
            return _get_quasi_diag_recursive(sort_ix, linkage, n)

        def _get_quasi_diag_recursive(sort_ix, linkage, n):
            out = []
            for ix in sort_ix:
                if ix >= n:
                    left = int(linkage[ix - n, 0])
                    right = int(linkage[ix - n, 1])
                    out.extend(_get_quasi_diag_recursive([left, right], linkage, n))
                else:
                    out.append(ix)
            return out

        def get_cluster_variance(cov, cluster_items):
            sub_cov = cov.loc[cluster_items, cluster_items]
            inv_diag = 1. / np.diag(sub_cov)
            weights = inv_diag / np.sum(inv_diag)
            return np.dot(weights, np.dot(sub_cov, weights))

        def recursive_bisection(cov, sorted_assets):
            weights = pd.Series(1.0, index=sorted_assets, dtype=float)
            clusters = [sorted_assets]
            while clusters:
                new_clusters = []
                for cluster in clusters:
                    if len(cluster) <= 1:
                        continue
                    split = int(len(cluster) / 2)
                    cluster1 = cluster[:split]
                    cluster2 = cluster[split:]
                    var1 = get_cluster_variance(cov, cluster1)
                    var2 = get_cluster_variance(cov, cluster2)
                    alpha = 1 - var1 / (var1 + var2)
                    weights.loc[cluster1] *= alpha
                    weights.loc[cluster2] *= (1 - alpha)
                    new_clusters += [cluster1, cluster2]
                clusters = new_clusters
            return weights

        cov = compute_robust_covariance(daily_returns)
        corr = daily_returns.corr()
        dist = compute_distance_matrix(corr)
        dist_condensed = squareform(dist.values, checks=False)
        linkage = sch.linkage(dist_condensed, method=method)

        sorted_indices = get_quasi_diag(linkage)
        sorted_tickers = [corr.index[i] for i in sorted_indices]
        cov_sorted = cov.loc[sorted_tickers, sorted_tickers]

        hrp_weights = recursive_bisection(cov_sorted, sorted_tickers)
        hrp_weights /= hrp_weights.sum()

        if max_w is not None:
            hrp_weights = hrp_weights.clip(upper=max_w)
            hrp_weights /= hrp_weights.sum()

        print("HRP portfolio weights:\n")
        for tck, wgt in hrp_weights.items():
            print(f"{tck}: {wgt:.4f}")

        print("\nFinished HRP run successfully.\n")

    except Exception as e:
        print(f"Error during HRP run: {e}")
    finally:
        sys.stdout = old_stdout

    return log_buffer.getvalue()

###############################################################################
# REPORTER AGENT => returns JSON with "Decision" and "Explanation"
###############################################################################
def reporter_agent(user_question: str, chatbot_output: str, td3_output: str, hrp_output: str) -> str:
    """
    We'll produce a JSON with fields "Decision" (Complete or Rerun) 
    and "Explanation" (the previous style's logic).
    """
    prompt = f"""
You are the Reporter/Recommender Agent.

We have the user question: {user_question}
We have the chatbot output: {chatbot_output}
We have the TD3 output: {td3_output}
We have the HRP output: {hrp_output}

We want a JSON with exactly two keys:
"Decision": either "Complete" if the simulation results appear to meet user demands, or "Rerun" if not
"Explanation": containing your reporter text

No extra keys. 
No extra text. 
Just valid JSON.
"""
    result = stream_llm_call(prompt)
    return sanitize_control_chars(result)

###############################################################################
# TD3 THREAD
###############################################################################
def run_td3_agent_thread(out_queue, hyper_json):
    import json
    class QueueStream:
        def __init__(self, q):
            self.q = q
        def write(self, msg):
            if msg.strip():
                self.q.put(msg if msg.endswith("\n") else msg + "\n")
        def flush(self):
            pass

    original_stdout = sys.stdout
    sys.stdout = QueueStream(out_queue)
    try:
        try:
            params = json.loads(hyper_json)
            actor_lr = float(params.get("actor_lr", 1e-4))
            critic_lr = float(params.get("critic_lr", 1e-3))
            gamma = float(params.get("gamma", 0.99))
            tau = float(params.get("tau", 0.005))
        except:
            actor_lr = 1e-4
            critic_lr = 1e-3
            gamma = 0.99
            tau = 0.005

        class EnhancedTradingEnv(AdvancedMultiAssetTradingEnv):
            def step(self, action):
                current_idx = self.current_step
                self.current_step += 1
                done = (self.current_step >= len(self.df) - 1)

                if current_idx > 0:
                    prev_row = self.df.iloc[current_idx - 1]
                    row = self.df.iloc[current_idx]
                    prev_spy = float(prev_row['Close_spy_SPY'])
                    cur_spy = float(row['Close_spy_SPY'])
                    prev_qqq = float(prev_row['Close_qqq_QQQ'])
                    cur_qqq = float(row['Close_qqq_QQQ'])
                    ret_spy = (cur_spy / prev_spy) - 1.0
                    ret_qqq = (cur_qqq / prev_qqq) - 1.0
                else:
                    ret_spy = 0.0
                    ret_qqq = 0.0

                w_spy = float(action[0])
                w_qqq = float(action[1])
                w_spy = max(0.0, min(w_spy, 1.0))
                w_qqq = max(0.0, min(w_qqq, 1.0))

                # small daily return approach
                day_return = (w_spy * ret_spy) + (w_qqq * ret_qqq)
                self.portfolio_value *= (1.0 + day_return)
                self.alloc_spy = w_spy
                self.alloc_qqq = w_qqq

                reward = day_return * 1000.0
                next_state = self._get_state()
                return next_state, float(reward), done, {}

        print("Environment initialized. Ready for training.\n")
        env = EnhancedTradingEnv(df_merged)
        env.reset()
        env.render()

        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]

        class TD3Actor(nn.Module):
            def __init__(self, state_dim, action_dim):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(state_dim, 256),
                    nn.ReLU(),
                    nn.Linear(256, 256),
                    nn.ReLU(),
                    nn.Linear(256, action_dim),
                    nn.Sigmoid()
                )
            def forward(self, x):
                return self.net(x)

        class TD3Critic(nn.Module):
            def __init__(self, state_dim, action_dim):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(state_dim + action_dim, 256),
                    nn.ReLU(),
                    nn.Linear(256, 256),
                    nn.ReLU(),
                    nn.Linear(256, 1)
                )
            def forward(self, state, action):
                return self.net(torch.cat([state, action], dim=1))

        class ReplayBuffer:
            def __init__(self, capacity):
                self.buffer = deque(maxlen=capacity)
            def push(self, state, action, reward, next_state, done):
                self.buffer.append((state, action, reward, next_state, done))
            def sample(self, batch_size):
                batch = random.sample(self.buffer, batch_size)
                state, action, reward, next_state, done = map(np.stack, zip(*batch))
                return state, action, reward, next_state, done
            def __len__(self):
                return len(self.buffer)

        class TD3Agent:
            def __init__(
                self,
                state_dim,
                action_dim,
                actor_lr=actor_lr,
                critic_lr=critic_lr,
                gamma=gamma,
                tau=tau,
                policy_noise=0.2,
                noise_clip=0.5,
                policy_freq=2,
                device='cpu'
            ):
                self.actor = TD3Actor(state_dim, action_dim).to(device)
                self.actor_target = TD3Actor(state_dim, action_dim).to(device)
                self.actor_target.load_state_dict(self.actor.state_dict())

                self.critic1 = TD3Critic(state_dim, action_dim).to(device)
                self.critic2 = TD3Critic(state_dim, action_dim).to(device)
                self.critic1_target = TD3Critic(state_dim, action_dim).to(device)
                self.critic2_target = TD3Critic(state_dim, action_dim).to(device)
                self.critic1_target.load_state_dict(self.critic1.state_dict())
                self.critic2_target.load_state_dict(self.critic2.state_dict())

                self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
                self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=critic_lr)
                self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=critic_lr)

                self.replay_buffer = ReplayBuffer(1_000_000)
                self.gamma = gamma
                self.tau = tau
                self.policy_noise = policy_noise
                self.noise_clip = noise_clip
                self.policy_freq = policy_freq
                self.device = device

                self.total_it = 0

            def select_action(self, state):
                # Add random exploration so the agent doesn't stagnate
                s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                base_action = self.actor(s).cpu().data.numpy().flatten()
                noise = np.random.normal(0, 0.1, size=base_action.shape)
                action = np.clip(base_action + noise, 0, 1)
                return action

            def update(self, batch_size):
                if len(self.replay_buffer) < batch_size:
                    return

                self.total_it += 1
                state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)
                state = torch.FloatTensor(state).to(self.device)
                action = torch.FloatTensor(action).to(self.device)
                reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
                next_state = torch.FloatTensor(next_state).to(self.device)
                done = torch.FloatTensor(done).unsqueeze(1).to(self.device)

                with torch.no_grad():
                    noise = (
                        torch.randn_like(action) * self.policy_noise
                    ).clamp(-self.noise_clip, self.noise_clip)
                    next_action = self.actor_target(next_state)
                    next_action = (next_action + noise).clamp(0, 1)

                    target_q1 = self.critic1_target(next_state, next_action)
                    target_q2 = self.critic2_target(next_state, next_action)
                    target_q = torch.min(target_q1, target_q2)
                    target_q = reward + (1 - done) * self.gamma * target_q

                current_q1 = self.critic1(state, action)
                loss_q1 = nn.MSELoss()(current_q1, target_q)
                self.critic1_optimizer.zero_grad()
                loss_q1.backward()
                self.critic1_optimizer.step()

                current_q2 = self.critic2(state, action)
                loss_q2 = nn.MSELoss()(current_q2, target_q)
                self.critic2_optimizer.zero_grad()
                loss_q2.backward()
                self.critic2_optimizer.step()

                if self.total_it % self.policy_freq == 0:
                    actor_loss = -self.critic1(state, self.actor(state)).mean()
                    self.actor_optimizer.zero_grad()
                    actor_loss.backward()
                    self.actor_optimizer.step()

                    for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                        target_param.data.copy_(self.tau*param.data + (1-self.tau)*target_param.data)
                    for param, target_param in zip(self.critic1.parameters(), self.critic1_target.parameters()):
                        target_param.data.copy_(self.tau*param.data + (1-self.tau)*target_param.data)
                    for param, target_param in zip(self.critic2.parameters(), self.critic2_target.parameters()):
                        target_param.data.copy_(self.tau*param.data + (1-self.tau)*target_param.data)

        agent = TD3Agent(state_dim, action_dim)
        print(f"TD3 Agent created. Hyperparameters:\n actor_lr={actor_lr}, critic_lr={critic_lr}, gamma={gamma}, tau={tau}\n")

        rewards = []
        portfolio_values = []
        num_episodes = 50
        batch_size = 64
        max_steps = len(df_merged) - 1

        print("Training TD3 Agent...\n")
        for ep in range(num_episodes):
            state = env.reset()
            ep_reward = 0.0
            for _ in range(max_steps):
                action = agent.select_action(state)
                next_state, reward, done, _ = env.step(action)
                agent.replay_buffer.push(state, action, reward, next_state, float(done))
                state = next_state
                ep_reward += reward
                agent.update(batch_size)
                if done:
                    break

            rewards.append(ep_reward)
            smoothed = float(np.mean(rewards[-10:]))
            portfolio_values.append(float(env.portfolio_value))

            print(f"Episode {ep+1}/{num_episodes}, Reward: {ep_reward:.4f}, Smoothed: {smoothed:.4f}, Final Portfolio: ${env.portfolio_value:.2f}\n")
            metrics_update = json.dumps({
                "smoothed_rewards": [float(x) for x in rewards],
                "portfolio_values": [float(x) for x in portfolio_values]
            })
            print(f"METRICS_UPDATE: {metrics_update}\n")

        print("\nTD3 Training complete.\n")

    except Exception as e:
        print(f"Error during TD3 run: {e}\n")
    finally:
        sys.stdout = original_stdout
        out_queue.put(None)

# ------------------------------ FIX STARTS HERE ------------------------------
def ensure_session_state():
    """
    Initializes any session_state attributes that might be missing.
    """
    if "workflow_started" not in st.session_state:
        st.session_state.workflow_started = False
    if "workflow_complete" not in st.session_state:
        st.session_state.workflow_complete = False
    if "hrp_params_decided" not in st.session_state:
        st.session_state.hrp_params_decided = False
    if "hrp_done" not in st.session_state:
        st.session_state.hrp_done = False
    if "hyper_done" not in st.session_state:
        st.session_state.hyper_done = False
    if "td3_done" not in st.session_state:
        st.session_state.td3_done = False
    if "td3_run" not in st.session_state:
        st.session_state.td3_run = False
    if "td3_queue" not in st.session_state:
        st.session_state.td3_queue = None
    if "td3_thread" not in st.session_state:
        st.session_state.td3_thread = None
    if "td3_output" not in st.session_state:
        st.session_state.td3_output = ""
    if "td3_lines" not in st.session_state:
        st.session_state.td3_lines = []
    if "reporter_json" not in st.session_state:
        st.session_state.reporter_json = ""
    if "conversation" not in st.session_state:
        st.session_state.conversation = []
    if "pdf_text" not in st.session_state:
        st.session_state.pdf_text = ""
    if "planner_output" not in st.session_state:
        st.session_state.planner_output = ""
    if "subtasks" not in st.session_state:
        st.session_state.subtasks = []
    if "subtask_index" not in st.session_state:
        st.session_state.subtask_index = 0
    if "subtask_answers" not in st.session_state:
        st.session_state.subtask_answers = []
    if "final_answer" not in st.session_state:
        st.session_state.final_answer = ""
    if "hrp_json" not in st.session_state:
        st.session_state.hrp_json = ""
    if "hrp_dict" not in st.session_state:
        st.session_state.hrp_dict = {}
    if "hrp_output" not in st.session_state:
        st.session_state.hrp_output = ""
    if "user_question_input" not in st.session_state:
        st.session_state.user_question_input = ""
    # Ensure hyper_json is also initialized
    if "hyper_json" not in st.session_state:
        st.session_state.hyper_json = ""
# ------------------------------ FIX ENDS HERE -------------------------------

def main():
    ensure_session_state()

    st.title("Agentic Hedge Fund Chatbot")

    def show_truncated_text(idx, full_text, max_chars=300):
        expand_key = f"expand_{idx}"
        if "expanded_items" not in st.session_state:
            st.session_state.expanded_items = {}
        expanded = st.session_state.expanded_items.get(expand_key, False)
        if len(full_text) <= max_chars:
            st.write(full_text)
            return
        if not expanded:
            truncated = full_text[:max_chars] + "..."
            st.write(truncated)
            if st.button("Expand", key=f"expand_btn_{idx}"):
                st.session_state.expanded_items[expand_key] = True
                st.rerun()
        else:
            st.write(full_text)
            if st.button("Collapse", key=f"collapse_btn_{idx}"):
                st.session_state.expanded_items[expand_key] = False
                st.rerun()

    col1, col2 = st.columns([2, 1])

    # ================ LEFT COLUMN => Chatbot subtask approach ================
    with col1:
        if not st.session_state.workflow_started:
            with st.form("init_form", clear_on_submit=True):
                uploaded_pdf = st.file_uploader("Upload PDF", type=["pdf"])
                user_question = st.text_input("Enter your question here")
                submit_init = st.form_submit_button("Run Chatbot")
                if submit_init:
                    if uploaded_pdf is None:
                        st.error("Please upload a PDF.")
                    elif not user_question.strip():
                        st.error("Please enter a question.")
                    else:
                        with st.spinner("Running Chatbot..."):
                            temp_pdf_path = "temp_uploaded_file.pdf"
                            with open(temp_pdf_path, "wb") as f:
                                f.write(uploaded_pdf.read())
                            raw_pdf_text = read_pdf_text(temp_pdf_path)
                            sanitized_pdf_text = sanitize_control_chars(raw_pdf_text)
                            st.session_state.pdf_text = sanitized_pdf_text

                            init_state: AgentWorkflowState = {
                                "pdf_path": temp_pdf_path,
                                "user_question": user_question,
                                "pdf_text": st.session_state.pdf_text,
                                "planner_output": "",
                                "execution_result": {},
                                "final_answer": ""
                            }
                            planner_state = compliance_planner_node(init_state)
                            st.session_state.planner_output = planner_state["planner_output"]
                            try:
                                plan_dict = json.loads(st.session_state.planner_output)
                                st.session_state.subtasks = plan_dict.get("subtasks", [])
                                st.session_state.subtask_answers = [None] * len(st.session_state.subtasks)
                            except Exception as e:
                                st.session_state.conversation.append(f"Planner error: {e}")
                            st.session_state.workflow_started = True
                            st.session_state.user_question_input = user_question
                            st.rerun()
        else:
            st.write("### Conversation Output")
            for i, line in enumerate(st.session_state.conversation):
                show_truncated_text(i, line, 300)
                if "[Subtask #" in line:
                    st.markdown(
                        "<strong style='display:block; width:100%; border-bottom:4px solid black; margin:10px 0;'></strong>",
                        unsafe_allow_html=True
                    )
            if st.session_state.subtask_index < len(st.session_state.subtasks):
                current = st.session_state.subtasks[st.session_state.subtask_index]
                subtask_type = current["type"].upper()

                if subtask_type == "LLM":
                    context_str = ""
                    for i, ans in enumerate(st.session_state.subtask_answers):
                        if ans is not None:
                            context_str += (
                                f"[Subtask #{i+1} - {ans['type']}]\n"
                                f"Task: {ans['task']}\n\n"
                                f"Answer: {ans['answer']}\n\n"
                            )
                    prompt_llm = f"""
You are the Executer Agent. 
PDF content:
{st.session_state.pdf_text}

Context:
{context_str}

Now execute subtask:
Task: {current['task']}

If more user info needed => "INSUFFICIENT: <questions>"
Else answer.
"""
                    with st.spinner("Executing LLM subtask..."):
                        llm_response = stream_llm_call(prompt_llm).strip()

                    new_line = (
                        f"[Subtask #{st.session_state.subtask_index+1} - LLM]\n"
                        f"Task: {current['task']}\n\n"
                        f"Answer: {llm_response}"
                    )
                    st.session_state.conversation.append(new_line)
                    st.session_state.subtask_answers[st.session_state.subtask_index] = {
                        "type": "LLM",
                        "task": current["task"],
                        "answer": llm_response
                    }
                    st.session_state.subtask_index += 1
                    st.rerun()

                elif subtask_type == "USER":
                    question_text = rewrite_subtask_as_question(current["task"])
                    st.write(f"[Subtask #{st.session_state.subtask_index+1} - USER]\n{question_text}\n")
                    user_response = st.text_input("Your Answer:", key=f"user_ans_{st.session_state.subtask_index}")
                    if st.button("Submit Answer", key=f"submit_{st.session_state.subtask_index}"):
                        if not user_response.strip():
                            st.error("Please enter an answer.")
                        else:
                            new_line = (
                                f"[Subtask #{st.session_state.subtask_index+1} - USER]\n"
                                f"Task: {current['task']}\n\n"
                                f"Answer: {user_response}"
                            )
                            st.session_state.conversation.append(new_line)
                            st.session_state.subtask_answers[st.session_state.subtask_index] = {
                                "type": "USER",
                                "task": current["task"],
                                "answer": user_response
                            }
                            st.session_state.subtask_index += 1
                            st.rerun()
            else:
                if not st.session_state.workflow_complete:
                    final_state: AgentWorkflowState = {
                        "pdf_path": "",
                        "user_question": st.session_state.user_question_input,
                        "pdf_text": st.session_state.pdf_text,
                        "planner_output": st.session_state.planner_output,
                        "execution_result": {"subtask_answers": st.session_state.subtask_answers},
                        "final_answer": ""
                    }
                    workflow = build_workflow().compile()
                    final_state = workflow.invoke(final_state)
                    st.session_state.final_answer = final_state["final_answer"]
                    st.session_state.workflow_complete = True
                    st.rerun()
                else:
                    st.write("### Final Answer")
                    st.write(st.session_state.final_answer)

    # ================ RIGHT COLUMN => HRP param decider, HRP, TD3, Reporter ===============
    with col2:
        hrp_param_btn = st.button(
            "Decide HRP Parameters (From Chatbot Answer)",
            disabled=(not st.session_state.workflow_complete),
            help="Use the chatbot final answer to produce HRP param JSON."
        )
        if hrp_param_btn:
            with st.spinner("Deciding HRP parameters..."):
                hrp_dict = hrp_param_decider(st.session_state.final_answer)
                st.session_state.hrp_dict = hrp_dict
                st.session_state.hrp_json = json.dumps(hrp_dict, indent=4)
                if hrp_dict:
                    st.session_state.hrp_params_decided = True
                else:
                    st.session_state.hrp_params_decided = False
                    st.warning("LLM gave invalid or empty HRP param JSON. Using fallback defaults if you proceed.")
            st.success("HRP parameters decided. Now you can run HRP.")

        if st.session_state.hrp_json:
            st.write("### Decided HRP Parameters (JSON)")
            st.text_area("HRP Params", st.session_state.hrp_json, height=250)

        run_hrp_btn = st.button(
            "Run Hierarchical Risk Parity",
            disabled=(
                not st.session_state.workflow_complete
                or not st.session_state.get("hrp_params_decided", False)
            ),
            help="Becomes active once chatbot is done and HRP params are decided."
        )
        if run_hrp_btn:
            with st.spinner("Running HRP..."):
                hrp_logs = run_hrp_inline(
                    st.session_state.final_answer,
                    st.session_state.hrp_dict
                )
                st.session_state.hrp_output = hrp_logs
                st.session_state.hrp_done = True
            st.success("HRP run complete.")

        if st.session_state.hrp_done:
            st.write("### HRP Agent Output")
            st.text_area("HRP Log", st.session_state.hrp_output, height=250)

        run_hyper_btn = st.button(
            "Run Hyperparameter Decider",
            disabled=(not st.session_state.hrp_done),
            help="Active once HRP is done."
        )
        if run_hyper_btn:
            with st.spinner("Deciding hyperparameters..."):
                st.session_state.hyper_json = hyperparameter_decider(st.session_state.hrp_output)
                st.session_state.hyper_done = True
            st.success("Hyperparameters decided.")

        if st.session_state.hyper_json:
            st.write("### Decided Hyperparameters (JSON)")
            st.text_area("Decided Hyperparameters", st.session_state.hyper_json, height=250)

        if "td3_lines" not in st.session_state:
            st.session_state.td3_lines = []

        run_td3_btn = st.button(
            "Run TD3 Trading Simulation",
            disabled=(not st.session_state.hyper_done),
            help="Active once hyperparameters are decided."
        )
        if run_td3_btn:
            if not st.session_state.td3_run and not st.session_state.td3_done:
                st.session_state.td3_lines = []
                st.session_state.td3_run = True
                st.session_state.td3_output = ""
                st.session_state.td3_queue = queue.Queue()
                st.session_state.td3_thread = threading.Thread(
                    target=run_td3_agent_thread,
                    args=(st.session_state.td3_queue, st.session_state.hyper_json)
                )
                st.session_state.td3_thread.start()

        if st.session_state.td3_run:
            st.write("### TD3 Trading Agent Output")
            td3_output_container = st.empty()
            chart_rewards_container = st.empty()
            chart_portfolio_container = st.empty()

            while True:
                while st.session_state.td3_queue and not st.session_state.td3_queue.empty():
                    msg = st.session_state.td3_queue.get_nowait()
                    if msg is None:
                        st.session_state.td3_run = False
                        st.session_state.td3_done = True
                        st.session_state.td3_thread = None
                        break
                    if msg.startswith("METRICS_UPDATE:"):
                        try:
                            metrics_json = msg.split("METRICS_UPDATE:")[1].strip()
                            metrics = json.loads(metrics_json)
                            current_rewards = metrics.get("smoothed_rewards", [])
                            current_portfolio = metrics.get("portfolio_values", [])
                            df_rewards = pd.DataFrame({
                                "Episode": list(range(1, len(current_rewards)+1)),
                                "Smoothed Reward": current_rewards
                            })
                            df_portfolio = pd.DataFrame({
                                "Episode": list(range(1, len(current_portfolio)+1)),
                                "Final Portfolio": current_portfolio
                            })
                            with chart_rewards_container.container():
                                st.markdown("**Chart 1: Smoothed Reward over Episodes**")
                                chart_r = alt.Chart(df_rewards).mark_line().encode(
                                    x=alt.X("Episode:Q", title="Episode"),
                                    y=alt.Y("Smoothed Reward:Q", title="Smoothed Reward")
                                ).properties(width=350, height=200)
                                st.altair_chart(chart_r, use_container_width=True)

                            with chart_portfolio_container.container():
                                st.markdown("**Chart 2: Final Portfolio Value over Episodes**")
                                chart_p = alt.Chart(df_portfolio).mark_line().encode(
                                    x=alt.X("Episode:Q", title="Episode"),
                                    y=alt.Y("Final Portfolio:Q", title="Final Portfolio Value")
                                ).properties(width=350, height=200)
                                st.altair_chart(chart_p, use_container_width=True)
                        except Exception as e:
                            st.session_state.td3_lines.append(f"Error parsing metrics: {e}\n")
                    else:
                        st.session_state.td3_lines.append(msg)
                        st.session_state.td3_output += msg

                td3_output_html = "<br>".join(st.session_state.td3_lines)
                td3_output_container.markdown(
                    f"<div style='width:100%; height:400px; overflow-y:auto; white-space:pre-wrap; "
                    f"font-family: monospace; font-size: 10px;'>{td3_output_html}</div>",
                    unsafe_allow_html=True
                )
                if not st.session_state.td3_run:
                    break
                time.sleep(0.5)

        run_report_btn = st.button(
            "Run Reporter/Recommender Agent",
            disabled=(not st.session_state.td3_done),
            help="Active once TD3 is finished."
        )
        if run_report_btn:
            if not st.session_state.td3_output:
                st.error("No TD3 output found. Please run the simulation first.")
            else:
                with st.spinner("Running Reporter Agent..."):
                    reporter_json = reporter_agent(
                        st.session_state.get("user_question_input", ""),
                        st.session_state.final_answer,
                        st.session_state.td3_output,
                        st.session_state.hrp_output
                    )
                    st.session_state.reporter_json = reporter_json
                st.success("Reporter/Recommender Agent complete.")

        if st.session_state.get("reporter_json"):
            st.write("### Reporter/Recommender Output")
            st.text_area("Reporter/Recommender JSON", st.session_state.reporter_json, height=300)

if __name__ == "__main__":
    main()