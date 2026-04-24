import requests
from typing import TypedDict, List, Optional
from langgraph.graph import StateGraph, END
from ollama import Client

# ── 1. State ──────────────────────────────────────────────
# 整張圖共享這個物件，每個 node 讀它、更新它

class SimulationState(TypedDict):
    personas: List[dict]           # 所有角色的 persona 資料
    topic: str                     # 本次會議主題
    messages: List[dict]           # 對話歷史 {"speaker", "target", "content"}
    current_speaker: str           # 這輪發問的人
    current_responder: str         # 這輪回應的人
    round: int                     # 目前第幾輪
    max_rounds: int                # 最多幾輪
    should_end: bool               # 是否應該結束
    summary: Optional[str]         # 最終總結

# ── 2. Nodes ──────────────────────────────────────────────
# 每個 node 就是一個普通函式：吃 state，吐出要更新的欄位

def decide_speaker(state: SimulationState) -> dict:
    """決定這輪由誰發問"""

    names = [p["name"] for p in state["personas"]]
    history_text = _format_history(state["messages"])

    prompt = f"""
你是一個會議主持人，請根據以下對話歷史，決定下一個應該發言提問的人。

參與者：{", ".join(names)}

對話歷史：
{history_text}

規則：
- 避免同一個人連續發言超過兩次
- 優先選擇還沒發言或發言較少的人
- 只回答名字，不要加任何解釋

下一個發言者："""

    speaker = _call_llm(prompt).strip()
    print(f"決定下一個發問者是：{speaker}")
    return {"current_speaker": speaker}


def decide_responder(state: SimulationState) -> dict:
    """決定誰來回應這輪的問題"""

    names = [p["name"] for p in state["personas"]
             if p["name"] != state["current_speaker"]]
    history_text = _format_history(state["messages"])

    prompt = f"""
根據以下對話歷史，決定最適合回應 {state["current_speaker"]} 的人是誰。

參與者（不含發問者）：{", ".join(names)}

對話歷史：
{history_text}

規則：
- 選擇跟這個話題最相關的人
- 或選擇最久沒有發言的人
- 只回答名字，不要加任何解釋

回應者："""

    responder = _call_llm(prompt).strip()
    print(f"決定回應者是：{responder}")
    return {"current_responder": responder}


def generate_dialogue(state: SimulationState) -> dict:
    """產生這輪的對話內容"""

    speaker_persona = load_persona(f"persona/{state['current_speaker']}.md")
    responder_persona = load_persona(f"persona/{state['current_responder']}.md")
    history_text = _format_history(state["messages"])

    # 發問者說話
    speaker_prompt = f"""
你是 {state["current_speaker"]}，以下是你的人物設定：
{speaker_persona}

會議主題：{state["topic"]}
目前是第 {state["round"]} 輪對話。

對話歷史：
{history_text}

請根據你的個性和研究進度，向 {state["current_responder"]} 提出一個問題或發表看法。
直接說話，不要加名字前綴，不要解釋你為什麼這樣說。
"""
    speaker_content = _call_llm(speaker_prompt).strip()
    print(f"{state['current_speaker']} 說：{speaker_content}")

    # 回應者說話
    responder_prompt = f"""
你是 {state["current_responder"]}，以下是你的人物設定：
{responder_persona}

會議主題：{state["topic"]}

對話歷史：
{history_text}

{state["current_speaker"]} 剛說了：「{speaker_content}」

請根據你的個性回應。
直接說話，不要加名字前綴，不要解釋你為什麼這樣說。
回答請用繁體中文。
"""
    responder_content = _call_llm(responder_prompt).strip()
    print(f"{state['current_responder']} 說：{responder_content}")
    new_messages = state["messages"] + [
        {
            "speaker": state["current_speaker"],
            "target": state["current_responder"],
            "content": speaker_content
        },
        {
            "speaker": state["current_responder"],
            "target": state["current_speaker"],
            "content": responder_content
        }
    ]

    return {
        "messages": new_messages,
        "round": state["round"] + 1
    }


def check_should_end(state: SimulationState) -> dict:
    """讓 LLM 判斷對話是否應該結束"""

    # 強制結束條件
    if state["round"] >= state["max_rounds"]:
        return {"should_end": True}

    history_text = _format_history(state["messages"])

    prompt = f"""
以下是一場實驗室會議的對話記錄：

{history_text}

請判斷這場對話是否已經可以結束？
結束的條件是：話題已經有了結論、或討論陷入明顯的僵局、或所有人都表達過意見了。

只回答 yes 或 no。"""

    answer = _call_llm(prompt).strip().lower()
    return {"should_end": answer == "yes"}


def summarize(state: SimulationState) -> dict:
    """總結本次實驗"""

    history_text = _format_history(state["messages"])
    names = [p["name"] for p in state["personas"]]

    prompt = f"""
以下是一場實驗室會議的完整對話記錄：

主題：{state["topic"]}
參與者：{", ".join(names)}

{history_text}

請根據對話內容，產生以下格式的總結，回答請用繁體中文：

## 會議總結
（整體討論了什麼，有沒有達成共識）

## 各人表現
（每個人在這次會議中的參與狀況和立場）

## 待解決的問題
（討論後仍然懸而未決的事項）
"""

    summary = _call_llm(prompt).strip()
    print(summary)
    return {"summary": summary}


# ── 3. Conditional Edge ───────────────────────────────────
# 這個函式決定 check_should_end 之後要去哪

def route_after_check(state: SimulationState) -> str:
    if state["should_end"]:
        return "summarize"
    else:
        return "decide_speaker"


# ── 4. 組裝 Graph ─────────────────────────────────────────

def build_graph() -> StateGraph:
    builder = StateGraph(SimulationState)

    # 加入所有 nodes
    builder.add_node("decide_speaker", decide_speaker)
    builder.add_node("decide_responder", decide_responder)
    builder.add_node("generate_dialogue", generate_dialogue)
    builder.add_node("check_should_end", check_should_end)
    builder.add_node("summarize", summarize)

    # 設定起點
    builder.set_entry_point("decide_speaker")

    # 固定邊：流程的主幹
    builder.add_edge("decide_speaker", "decide_responder")
    builder.add_edge("decide_responder", "generate_dialogue")
    builder.add_edge("generate_dialogue", "check_should_end")
    builder.add_edge("summarize", END)

    # 條件邊：check 之後要繼續還是結束
    builder.add_conditional_edges(
        "check_should_end",
        route_after_check,
        {
            "decide_speaker": "decide_speaker",
            "summarize": "summarize"
        }
    )

    return builder.compile()


# ── 5. 工具函式 ───────────────────────────────────────────

def _call_llm(prompt: str) -> str:
    """呼叫你的本地 LLM，這裡換成你實際用的方式"""
    # 如果你用 ollama：
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": "qwen2.5:7b-instruct", "prompt": prompt, "stream": False}
    )
    return response.json()["response"]


def _format_history(messages: List[dict]) -> str:
    if not messages:
        return "（尚無對話）"
    return "\n".join(
        f"{m['speaker']} 對 {m['target']} 說：{m['content']}"
        for m in messages
    )

def load_persona(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# ── 6. 執行入口 ───────────────────────────────────────────

if __name__ == "__main__":
    # 讀入你的 .md 檔案內容
    personas = [
        {"name": "老周", "persona_text": load_persona("persona/老周.md")},
        {"name": "維維", "persona_text": load_persona("persona/維維.md")},
        {"name": "TN",   "persona_text": load_persona("persona/TN.md")},
        {"name": "小諺", "persona_text": load_persona("persona/小諺.md")},
        {"name": "阿火", "persona_text": load_persona("persona/阿火.md")},
        {"name": "綜哥", "persona_text": load_persona("persona/綜哥.md")},
    ]

    graph = build_graph()

    result = graph.invoke({
        "personas": personas,
        "topic": "本週實驗室進度會議",
        "messages": [],
        "current_speaker": "",
        "current_responder": "",
        "round": 0,
        "max_rounds": 5,
        "should_end": False,
        "summary": None
    })

    print(result["summary"])
