from langchain.agents import create_agent
from dotenv import load_dotenv
from langchain.messages import HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
import config
import sqlite3
from tool_2 import *
load_dotenv()
from langchain_openai import ChatOpenAI
import os
from tool_3 import local_search

connection=sqlite3.connect(config.sqlite_path,check_same_thread=False)
checkpoint=SqliteSaver(connection)#自动存储数据和会话
checkpoint.setup()#checkpoint 对象的初始化准备方法

system_prompt="""你拥有两个工具：
1. local_search：本地知识库/数据库查询工具，查询个人项目、本地存储的资料，用于已知的存量信息；
2. search_tools：联网搜索工具，查询互联网实时、最新的外部信息。
工具调用规则，严格遵守：
1. 优先级判定：
   - 如果**用户明确指定使用某个工具**，强制遵循用户指定的工具执行，忽略自主判断逻辑；
     用户指定工具识别关键词：「查本地库/查知识库/本地查询」→ local_search；「联网搜、网上查、搜索互联网」→ search_tools。
   - 如果**用户没有指定工具**，由智能体自主判断：问题属于【本地知识库范围内的内容】，优先调用 local_search；
2. 仅当本地知识库返回：无相关资料/找不到信息，才允许调用联网搜索search_tools；
3. ❌ 禁止并行同时调用两个工具，不能一次性发起两个tool call；
4. 不允许预先联网；本地查询失败后，再考虑联网；
5. 回答优先采信本地知识库内容；本地无结果，才使用联网结果；
6. 如果两个工具信息冲突，优先使用本地知识库内容。
7.在回答的最后贴出参考依据，如果是本地资料，显示资料来源，不允许修改模型返回的‘检索文件来源以及页码’，必须原样返回。当调用联网搜索时，必须显式资料来源和网址，并且说明为联网搜索。
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
    tools=[local_search,search_tools],
    system_prompt=system_prompt,
    checkpointer=checkpoint,
)

messages=HumanMessage("BLEU 分数是多少？")

result=main_agent.invoke(
    {"messages":[messages]},
    config=config.config_thread
)
for i in result["messages"]:
    print(i.pretty_print())