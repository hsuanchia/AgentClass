import random, time
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Dict
from ollama import Client


# =========================
# 1️⃣ State 定義
# =========================

class Agent(TypedDict):
    name: str
    personality: str
    action: str
    target: str
    message: str
    memory: List[str]

class State(TypedDict):
    agents: List[Agent]
    messages: List[Dict]
    time: int
    event: str


# =========================
# 2️⃣ LLM Thinking
# =========================

def llm_decision(agent, event):
    actions = ["idle", "chat", "sleep"]

    if event == "meeting":
        return {
            "action": "chat",
            "target": "all",
            "message": f"{agent['name']}：我覺得這次會議很重要"
        }

    action = random.choice(actions)

    if action == "chat":
        target = random.choice(["阿火", "小諺", "庭哥"])
        message = llm_question(target=target, personality=agent["personality"], history=agent["memory"], name=agent["name"])
    else: 
        target = "" 
        message = "" 

    return { "action": action, "target": target, "message": message }

# =========================
# 3️⃣ Node: 更新環境
# =========================

def update_environment(state: State):
    new_time = state["time"] + 1

    if random.random() < 0.2:
        event = random.choice(["meeting", "break"])
        print(f"\n📢 Event 發生: {event}")
    else:
        event = None

    return {
        "time": new_time,
        "event": event
    }


# =========================
# 4️⃣ Node: Agent Thinking（唯一更新 agents）
# =========================

def agent_thinking(state: State):
    new_agents = []

    for agent in state["agents"]:
        decision = llm_decision(agent, state["event"])

        new_agent = {
            **agent,
            "action": decision["action"],
            "target": decision["target"],
            "message": decision["message"]
        }

        new_agents.append(new_agent)

    return {
        "agents": new_agents
    }


# =========================
# 5️⃣ Node: Agent Action（只處理 messages）
# =========================

def agent_action(state: State):
    new_messages = []
    new_agents = []

    agents_dict = {a["name"]: a for a in state["agents"]}

    speaking_agents = [
        a for a in state["agents"] if a["action"] == "chat"
    ][:2]

    for agent in state["agents"]:
        new_agents.append(agent.copy())

    for speaker in speaking_agents:
        target_name = speaker["target"]

        if target_name not in agents_dict:
            continue

        target_agent = agents_dict[target_name]

        # ⭐ 建立對話 context（超重要）
        history = target_agent["memory"][-5:]

        response = llm_answer(
            query=speaker["message"],
            personality=target_agent["personality"],
            history=history,
            name=target_agent["name"]
        )

        print(f"{speaker['name']} 問: {speaker['message']}")
        print(f"{target_name} 回: {response}\n")

        # ⭐ 更新 memory
        for agent in new_agents:
            if agent["name"] == speaker["name"]:
                agent["memory"].append(f"我對 {target_name} 說: {speaker['message']}")
            if agent["name"] == target_name:
                agent["memory"].append(f"{speaker['name']} 對我說: {speaker['message']}")
                agent["memory"].append(f"我回: {response}")

        new_messages.append({
            "from": speaker["name"],
            "to": target_name,
            "message": speaker["message"],
            "response": response
        })

    return {
        "messages": state["messages"] + new_messages,
        "agents": new_agents   # ⭐ 注意這裡（唯一更新）
    }

def llm_answer(query, personality, history, name):
    history_text = "\n".join(history)

    prompt = f"""
            你現在在模擬一個真實人類對話。

            你的名字是：{name}

            人格特質：
            {personality}

            過去對話紀錄：
            {history_text}

            請根據以上資訊，用自然口語回應對方。

            限制：
            1. 只能用繁體中文
            2. 回答簡短（1~2句）
            3. 要符合人格（不要變來變去）
            4. 不要解釋自己是AI

            對方說：
            {query}

            你的回應：
            """

    client = Client()

    response = client.chat(
        model="qwen:7b",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return response.message['content']

def llm_question(target, personality, history, name):
    history_text = "\n".join(history)

    prompt = f"""
            你現在在模擬一個真實人類對話。

            你的名字是：{name}

            人格特質：
            {personality}

            過去對話紀錄：
            {history_text}

            請根據以上資訊，用自然口語和對方交流。

            限制：
            1. 只能用繁體中文
            2. 回答簡短（1~2句）
            3. 要符合人格（不要變來變去）
            4. 不要解釋自己是AI

            交流對象是：
            {target}

            你的回應：
            """

    client = Client()

    response = client.chat(
        model="qwen:7b",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return response.message['content']

# =========================
# 6️⃣ 終止條件
# =========================

def should_continue(state: State):
    if state["time"] >= 5:
        return END
    return "agent_thinking"


# =========================
# 7️⃣ 建 Graph
# =========================

builder = StateGraph(State)

builder.add_node("update_environment", update_environment)
builder.add_node("agent_thinking", agent_thinking)
builder.add_node("agent_action", agent_action)

builder.set_entry_point("update_environment")

builder.add_edge("update_environment", "agent_thinking")
builder.add_edge("agent_thinking", "agent_action")
builder.add_edge("agent_action", "update_environment")

builder.add_conditional_edges(
    "update_environment",
    should_continue
)

graph = builder.compile()


# =========================
# 8️⃣ 初始化
# =========================

initial_state = {
    "agents": [
        {"name": "阿火", "personality": "內向", "action": "", "target": "", "message": "", "memory": []},
        {"name": "小諺", "personality": "外向", "action": "", "target": "", "message": "", "memory": []},
        {"name": "庭哥", "personality": "活潑", "action": "", "target": "", "message": "", "memory": []}
    ],
    "messages": [],
    "time": 0,
    "event": None
}


# =========================
# 🔟 執行
# =========================
result = graph.invoke(initial_state)

print("\n=== Simulation End ===")
print(result)