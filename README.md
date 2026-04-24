# AgentClass
Multi-Agent Social Simulation System
# Motivation
> 填坑之旅: 多年前和政大公行合作的案子，當年技術能力與時間有限，無力做完。現在有時間，也期望鍛鍊自己新的技術並將其重新規畫完成。 \
> 預期使用技術: LangGraph, MongoDB, React, FastAPI \
> Ref: [Out of One, Many: Using Language Models to Simulate Human Samples](https://arxiv.org/abs/2209.06899)

# Current
* LLM: QWEN-7B by Ollama-> 在我的 RTX 3060 Laptop GPU 勉強能跑(大約需要5.4G左右的獨顯記憶體)
    * 改用qwen2.5:7b-instruct -> 輸出結果比qwen:7b穩定很多
* 用Langgraph架起來了，設定好整體目標與流程
    * 給定人設以及討論目標
    * 初步希望agents之間互相討論5輪，每一輪由他們自己決定何時該終止討論
    * 目前模擬: 教授與研究生
        * 教授任務: 追蹤與推進每個研究生進度、主持每次會議
        * 研究生任務: 回應教授並推進自己研究進度

# To Do
* 每個人的memory處理
    * 用資料庫儲存(MongoDB)
    * 過往記憶應該要遺忘 -> 可以用摘要壓縮!! -> 每對話過幾輪之後，將先前的對話做個summary然後再儲存
* 人設和prompt必須要更詳細 -> 目前先用AI幫我生成
* 最後再包上FastAPI+React

# Package version
```bash
python==3.10.20
ollama==0.6.1
langgraph==1.1.8
```