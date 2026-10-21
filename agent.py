"""本智能体严格按照比赛要求进行本地检索，如果检索结果不符合问题，智能体将返回
‘本地资料库无相关内容’
若果想让智能体进行联网，只需要将工具too_2的函数放入智能体工具列表，斌且修改提示词即可。
"""
from langchain.agents import create_agent
from dotenv import load_dotenv
from langchain.messages import HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
import config
import sqlite3
from tool_1 import translate_paper
from tool_2 import *
load_dotenv()
from langchain_openai import ChatOpenAI
import os
from tool_3 import local_search

connection=sqlite3.connect(config.sqlite_path,check_same_thread=False)
checkpoint=SqliteSaver(connection)#自动存储数据和会话
checkpoint.setup()#checkpoint 对象的初始化准备方法

system_prompt = """你是科研文献助手，只依据本地已入库的文献作答，不用自己的先验知识补充细节。

【可用工具】
1. local_search    —— 查本地文献库。回答「论文里说了什么」「某个数字是多少」「为什么这样做」这类内容问题。
2. translate_paper —— 把本地文献按章节翻译成中文。只在用户明确要「译文」时使用。

【第一步：判断意图，只选一个工具】
判断标准是用户想拿到什么：
- 想要译文 → translate_paper
- 想要答案 → local_search

注意：「这段什么意思」「这篇讲了什么」「结论是什么」都是问内容，走 local_search，不要走翻译。
只有出现「翻译」「译成中文」「中文版」这类明确要求，才走 translate_paper。

【translate_paper 规则】
1. 一次只翻一节。section 传节标题，如 "Abstract"、"Conclusion"，不要留空。
   用户要整篇时，先说「我按节依次翻译」，然后一节一次、多次调用。
2. filename 必须和工具返回的文件名完全一致（如 1706.03762v1.pdf）。
   不确定文件名或节标题时，先调用工具，让报错信息告诉你可选值，不要编造。
3. 它只给译文，不给结论。用户既要译文又要总结时，先用它拿到译文，再基于译文作答。
4. 返回内容里的「【节标题】第 N 页」要保留。
5. 用户只想翻其中一两句时，直接用那几句原文给出中文即可，不要为此调用本工具。

【回答规则】
1. 每一条结论都要有 local_search 返回的原文支撑。原文里没有的数字、结论、页码，一律不写。
2. 本地查不到相关内容时，直接回答「库中没有相关资料」，不要猜、不要用通用知识作答。
3. 回答最后另起一行，原样输出工具返回的来源行，格式是：
   检索文件来源以及页码：1706.03762v1.pdf 第1页、第3页
   不要改写这一行、不要合并多个文件、不要只写其中几页、不要自己补页码。
4. 用中文，简洁专业。不要「我认为」「一般来说」「可能是」这类推测表述。
5. 一次只调用一个工具，不要并行发起多个工具调用。
"""
model=ChatOpenAI(
    model="qwen3.8-max",
    base_url=os.getenv("qwen_url"),
    api_key=os.getenv("DASHSCOPE_API_KEY"),
)
agent=create_agent(
    model
)

main_agent=create_agent(
    model,
    tools=[local_search,translate_paper],
    system_prompt=system_prompt,
    checkpointer=checkpoint,
)
if __name__=="__main__":
    messages=HumanMessage("BLEU 分数是多少？")

    result=main_agent.invoke(
        {"messages":[messages]},
        config=config.config_thread
    )
    for i in result["messages"]:
        print(i.pretty_print())