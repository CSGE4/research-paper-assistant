from langgraph.prebuilt import ToolRuntime
from rag import Rag

# 这是交给智能体的两个工具之一：本地检索。另一个是 tool_1 的翻译。
#
# 它的职责边界很硬：只查本地库，不联网、不推断、不总结。
# 答案由 rag.py 那条链生成，这里只负责把结构化的结果【拼成模型能读的一段文本】。

# 一轮问答最多检索几次。这个数以前只写在提示词里，模型可以不听：
# 实测同一轮搜了 22 次、跑满 250 秒才吐答案。现在搬进代码，到数就拒。
MAX_RETRIEVAL = 6

# 哪些工具算「一次检索」。web_search 也占额度，否则联网开着时它照样能一直试。
SEARCH_TOOL_NAMES = ("local_search", "web_search")


def count_retrieval(messages):
    """数一段消息里本轮已经完成的检索次数：往前数到最近一条用户提问为止。

    不用模块级计数器：Streamlit 每次交互都会把脚本从头重跑，全局变量既会
    串到别的会话，也不会在下一轮自动归零（4.3 那几条缺陷就是这么来的）。
    历史里本来就记着每次调用，数它最准，也不需要额外维护状态。
    """
    n = 0
    for msg in reversed(messages or []):
        if getattr(msg, "type", "") == "human":
            break
        if getattr(msg, "type", "") == "tool" and getattr(msg, "name", "") in SEARCH_TOOL_NAMES:
            n = n + 1
    return n


def used_retrieval(runtime):
    """工具内部用：本次调用之前，本轮已经检索了几次。"""
    return count_retrieval((runtime.state or {}).get("messages"))


def over_budget(runtime):
    """到上限就返回一句打发模型的话，没到返回 None。"""
    if used_retrieval(runtime) < MAX_RETRIEVAL:
        return None
    return ("本轮检索次数已达上限（%d 次），不能再调用检索工具。"
            "请直接用已经查到的内容作答；查不到的部分就说明没查到，不要换关键词再试。" % MAX_RETRIEVAL)


rag_instance = None
def local_search(query:str, runtime: ToolRuntime):
    """在本地知识库检索文档，仅查询本地资料，不会联网搜索
        Args:
            query: 用户的查询问题
        """
    stop = over_budget(runtime)
    if stop:
        return stop
    # 单例：一次问答里模型可能连着调好几次这个工具，
    # 每次都 new 一个 Rag 等于每次都重新打开向量库，几秒钟就没了。
    global rag_instance
    if rag_instance is None:
        rag_instance=Rag()
    result=rag_instance.get_answer(query)

    # 模型只认文本，所以把 dict 拼成字符串。
    # 来源行固定成「检索文件来源以及页码：...」这个格式，
    # 提示词里要求模型原样保留的正是这一行，格式一改规则就失效了。
    #
    # 注意 sources 是 rag.py 从检索结果算出来的，不是从模型回答里解析的——
    # 模型自己写的页码不可信，真正给用户看的来源由检索层说了算。
    sources=[]
    for name in result["sources"]:
        pages=sorted(result["sources"][name])
        page_text=""
        for p in pages:
            if page_text:
                page_text=page_text+"、"
            page_text=page_text+"第%d页" % p
        sources.append("%s %s" % (name, page_text))
    # 一块都没检索到时也要给一行，否则模型可能自己编一个来源出来
    if not sources:
        sources.append("无")

    return "%s\n\n检索文件来源以及页码：%s" % (result["answer"], "；".join(sources))
if __name__=="__main__":
    # 单独跑可以验一次工具返回：注意这里问的是库里没有的东西，正常应回"未提及"
    a=local_search("我的四级成绩是多少")
    print(a)
