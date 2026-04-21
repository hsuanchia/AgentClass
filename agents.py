import random, os
from typing import TypedDict, Dict, List
from langgraph.graph import StateGraph, END
from ollama import Client

# =========================
# State 定義
# =========================

class State(TypedDict):
    agents: List[str]
    personas: Dict[str, str]
    memory: Dict[str, Dict[str, List[str]]]

    active_pair: Dict
    last_message: Dict

    time: int

# =========================
# 工具
# =========================

def load_prompt(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def load_persona(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

# =========================
# Node 1: 選擇對話 pair
# =========================

def select_pair(state: State):
    q = random.choice(state["agents"])
    a = random.choice(state["agents"])

    while q == a:
        a = random.choice(state["agents"])

    return {
        "active_pair": {"q": q, "a": a}
    }

# =========================
# Node 2: 產生問題
# =========================

def generate_question(state: State):
    q = state["active_pair"]["q"]
    a = state["active_pair"]["a"]

    history = state["memory"][q][a][-3:]

    prompt = f"""
你的名字是：{q}

人格特質：
{state["personas"][q]}

你與 {a} 的過去對話：
{history}

請對 {a} 說一句話：
"""

    res = client.chat(
        model="qwen:7b",
        messages=[
            {"role": "system", "content": question_prompt},
            {"role": "user", "content": prompt}
        ]
    )

    return {
        "last_message": {
            "q": q,
            "a": a,
            "question": res.message["content"]
        }
    }

# =========================
# Node 3: 回答
# =========================

def generate_answer(state: State):
    q = state["last_message"]["q"]
    a = state["last_message"]["a"]
    question = state["last_message"]["question"]

    history = state["memory"][a][q][-3:]

    prompt = f"""
你的名字是：{a}

人格特質：
{state["personas"][a]}

你與 {q} 的過去對話：
{history}

對方說：
{question}

請回應：
"""

    res = client.chat(
        model="qwen:7b",
        messages=[
            {"role": "system", "content": answer_prompt},
            {"role": "user", "content": prompt}
        ]
    )

    print(f"{q} → {a}：{question}")
    print(f"{a} 回：{res.message['content']}\n")

    return {
        "last_message": {
            **state["last_message"],
            "answer": res.message["content"]
        }
    }

# =========================
# Node 4: 更新記憶
# =========================

def update_memory(state: State):
    q = state["last_message"]["q"]
    a = state["last_message"]["a"]
    question = state["last_message"]["question"]
    answer = state["last_message"]["answer"]

    memory = state["memory"]

    memory[q][a].append(f"我對 {a} 說：{question}")
    memory[a][q].append(f"{q} 對我說：{question}")
    memory[a][q].append(f"我回：{answer}")

    return {
        "memory": memory,
        "time": state["time"] + 1
    }

# =========================
# 停止條件
# =========================

def should_continue(state: State):
    if state["time"] >= 10:
        return END
    return "select_pair"

# =========================
# 主程式
# =========================

if __name__ == "__main__":
    client = Client()
    question_prompt = load_prompt("prompts/question.txt")
    answer_prompt = load_prompt("prompts/answer.txt")

    # 載入 persona
    personas = {}
    agents = []

    for file in os.listdir("persona/"):
        if file.endswith(".md"):
            name = file.split(".")[0]
            personas[name] = load_persona(f"persona/{file}")
            agents.append(name)

    # 初始化 memory（重點）
    memory = {
        a: {b: [] for b in agents if b != a}
        for a in agents
    }

    initial_state = {
        "agents": agents,
        "personas": personas,
        "memory": memory,
        "active_pair": {},
        "last_message": {},
        "time": 0
    }

    builder = StateGraph(State)

    builder.add_node("select_pair", select_pair)
    builder.add_node("generate_question", generate_question)
    builder.add_node("generate_answer", generate_answer)
    builder.add_node("update_memory", update_memory)

    builder.set_entry_point("select_pair")

    builder.add_edge("select_pair", "generate_question")
    builder.add_edge("generate_question", "generate_answer")
    builder.add_edge("generate_answer", "update_memory")

    builder.add_conditional_edges(
        "update_memory",
        should_continue
    )

    graph = builder.compile()

    graph.invoke(initial_state)