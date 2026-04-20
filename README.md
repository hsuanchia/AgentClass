# AgentClass
Multi-Agent Social Simulation System
# Motivation
> 填坑之旅: 多年前和政大公行合作的案子，當年技術能力與時間有限，無力做完。現在有時間，也期望鍛鍊自己新的技術並將其重新規畫完成。 \
> 預期使用技術: LangGraph, MongoDB, React, FastAPI \
> Ref: [Out of One, Many: Using Language Models to Simulate Human Samples](https://arxiv.org/abs/2209.06899)

# Current
* LLM: QWEN-7B by Ollama-> 在我的 RTX 3060 GPU 勉強能跑(大約需要5.4G左右的獨顯記憶體)
* 用Langgraph架起來了，初步可以互相聊天，但整體過程還需要優化

# To Do
* LangGraph整體流程優化
    * 互動更多
    * 狀態更新
    * Special event
* 每個人的memory處理
    * 用資料庫儲存? -> 那是不是personality也要用DB存比較好
    * 有些過往記憶應該要遺忘 -> 如何決定?
* 人設必須要更詳細
* 最後再包上FastAPI+React