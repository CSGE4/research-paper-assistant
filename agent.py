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
   工具返回的内容里出现「参考资料未提及」，也按这一条处理。
   只回这一句，不要列举检索关键词、调用了几次、换了哪些说法。
3. 回答最后另起一行写来源，格式是：
   检索文件来源以及页码：1706.03762v1.pdf 第1页、第8页
   页码只能取自工具返回的页码，并且只写真正支撑你结论的那几页：
   工具给了 6 页而你只用了 1 页，就只写那 1 页。不要自己推算或补页码。
   没有引用任何原文时（包括拒答），这些来源行整行不写，不要写「无」。
4. 用中文，简洁专业。不要「我认为」「一般来说」「可能是」这类推测表述。
5. 一次只调用一个工具，不要并行发起多个工具调用。
6. 不允许从文件名、编号推断事实。文件名里的「1706.03762」是 arXiv 编号，
   不能据此推断论文的发表时间，也不能把编号里的数字当成论文内容。
7. 不要做推断。原文没有用一句话直接说明的事实，就回答「库中没有明确说明」，
   不要用常识补全，也不要从参考文献条目的年份推断本文的信息。
"""
model=ChatOpenAI(
    model="qwen3.8-max",
    base_url=os.getenv("qwen_url"),
    api_key=os.getenv("DASHSCOPE_API_KEY"),
)
agent=create_agent(
    model
)

# 联网工具只是包一层：没配 TAVILY_API_KEY 时返回一句话让模型退回去用本地检索，
# 而不是让整个问答页报错。返回拼好的纯文本，模型只认文本。
def web_search(query: str):
    """联网检索公开资料。只有本地文献库确实查不到相关内容时才使用。
        Args:
            query: 要在互联网上检索的关键词或问题
        """
    tv = search_tools()
    if tv is None:
        return "联网搜索未开启（没有配置 TAVILY_API_KEY），只能依据本地检索结果作答。"
    data = tv.invoke({"query": query})
    items = []
    if isinstance(data, dict):
        items = data.get("results", [])
    if not items:
        return "联网搜索没有返回结果。"
    lines = []
    for i, it in enumerate(items, 1):
        content = it.get("content", "")
        if len(content) > 300:
            content = content[:300] + "…"
        lines.append("%d. %s\n%s\n%s" % (i, it.get("title", ""), it.get("url", ""), content))
    return "\n\n".join(lines)

# 联网时才追加的规则。必须显式覆盖前面「查不到就说库中没有相关资料」那条，
# 否则开关打开了，模型也只会继续回答"库中没有"，联网等于没开。
WEB_RULES = """
【本次已开启联网（多了一个 web_search 工具）】
1. 依然优先用 local_search。只有本地确实查不到时，才调用 web_search。
2. 用联网内容作答时，结论后面标「（来自互联网）」，并在最后一行加：
   联网来源：<链接>
3. 不能把联网内容说成本地文献里的内容，两边出处必须分开写。
4. 本地与联网冲突时，以本地文献为准，并说明存在冲突。
5. 前面「本地查不到就回答库中没有相关资料」那条，本次改成：先转 web_search，
   联网也没有结果时，再回答本地和网上都没有，用一句话说完，
   不要列举检索关键词、调用次数或检索过程。
"""

# 一个智能体，两套提示词。联网的值从本次调用传进来（app.py 里 invoke 的 context 参数），
# 所以同一个 agent 对象能同时服务联网和不联网两种会话，不用建两个。
from langchain.agents.middleware import dynamic_prompt
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
        return system_prompt + WEB_RULES
    return system_prompt

main_agent=create_agent(
    model,
    tools=[local_search,translate_paper,web_search],
    middleware=[pick_prompt],
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