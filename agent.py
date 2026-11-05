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
from search_budget import over_budget, MAX_RETRIEVAL
from tool_4 import list_documents
from promot import main_prompt, web_prompt

connection=sqlite3.connect(config.sqlite_path,check_same_thread=False)
checkpoint=SqliteSaver(connection)#自动存储数据和会话
checkpoint.setup()#checkpoint 对象的初始化准备方法

model=ChatOpenAI(
    model="qwen3.8-flash",
    base_url=os.getenv("qwen_url"),
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    # 实测同一个问题、同样的输入，有时 18 秒、有时 80 秒。
    # 差别不在计算量，在某个请求卡住干等——单次调用只要 2.8 秒，能等出 80 秒只能是没响应。
    # 45 秒足够覆盖正常最慢的一次（实测最长 11.1 秒）。
    # 重试次数必须是 0：原来的 2 次会把一次卡住放大成 45×3=135 秒，
    # 而这个智能体一次问答要调两三次模型，叠起来就是「检索文献中…」挂三分钟以上。
    # 卡住就让它快速失败，用户宁可看到一句报错，也不要看着转圈干等。
    timeout=45,
    max_retries=0,
    # qwen3.8 这一代默认带"思考"。实测同一段输入：82 秒里 4410 个输出 token，
    # 其中 4103 个是思考，正文只有 678 字。关掉之后 8.8 秒，正文反而 948 字。
    # 这里要的是照着检索到的原文复述，不需要它自己想，思考纯粹是烧钱拖时间。
    extra_body={"enable_thinking": False},
)
agent=create_agent(
    model
)

# 一个智能体，两套提示词。联网的值从本次调用传进来（app.py 里 invoke 的 context 参数），
# 所以同一个 agent 对象能同时服务联网和不联网两种会话，不用建两个。
from langchain.agents.middleware import dynamic_prompt, ToolCallLimitMiddleware
#导入dynamic_prompt装饰器，动态监测智能体状态，并根据当前状态切换提示词
"""
我--中间件--模型--工具--中间件--模型（此时回答）
中间件的次数 = 模型醒的次数 = 1 + 工具调用次数。
"""
@dynamic_prompt
def pick_prompt(request) -> str:
    use_web = False
    if request.runtime.context:
        use_web = request.runtime.context.get("use_web", False)
    if use_web:
        # 联网规则必须显式覆盖「查不到就说库中没有相关资料」那条，否则开关打开了，
        # 模型也只会继续回答"库中没有"，联网等于没开。
        return main_prompt + web_prompt
    return main_prompt

# 检索次数的硬上限，分两层，各管一件事：
#
# 1) 这个中间件管「停下来」。同一轮第 7 次调用 local_search 时直接结束这一轮，
#    而不是把拒绝文本丢回去。实测只回绝文本的话，模型会把拒绝当临时故障继续
#    重试：上限设成 2 时，2 次正常检索之后它又连调了 13 次被拒的调用，白烧 57 秒。
#    也试过在模型请求里摘掉工具、加 tool_choice="none"——日志确认工具确实摘掉了，
#    但这家 provider 照样往一个不存在的工具上发调用，挡不住，所以换成这个官方中间件。
#
# 2) 工具内部的 over_budget 管「查不了」。它按本地+联网合计计数，
#    是真正不让检索发生的那一道，中间件只是让过程早点结束。
#
#    两个检索工具都得挂。只挂 local_search 的时候，联网那条路是敞开的：模型被
#    over_budget 拒了之后会把拒绝当成临时故障，换个关键词再试，而这条路上没有
#    任何东西能结束这一轮——实测一轮点了 16 次、跑 118 秒，其中约十次是白等。
#    合计上限仍由 over_budget 把关（6 次），这里只管「到数就别再磨了」。
limit_local = ToolCallLimitMiddleware(
    tool_name="local_search",
    run_limit=MAX_RETRIEVAL,      # 按「一轮」算，不是按会话
    exit_behavior="end",
)
limit_web = ToolCallLimitMiddleware(
    tool_name="web_search",
    run_limit=MAX_RETRIEVAL,
    exit_behavior="end",
)

main_agent=create_agent(
    model,
    tools=[local_search,translate_paper,web_search,list_documents],
    middleware=[pick_prompt, limit_local, limit_web],
    checkpointer=checkpoint,
)
if __name__=="__main__":
    messages=HumanMessage("Transformer 是哪一年提出的？")

    result=main_agent.invoke(
        {"messages":[messages]},
        config=config.config_thread
    )
    for i in result["messages"]:
        print(i.pretty_print())