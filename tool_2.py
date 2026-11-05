"""Tavily 联网搜索（可选工具）。缺 key 时不要影响主流程。"""
import os
from dotenv import load_dotenv
# web_search 从 agent.py 搬过来之后，这两样必须跟着搬：
# ToolRuntime 在函数定义那一刻就要解析，over_budget 在真正联网前要调用。
from langgraph.prebuilt import ToolRuntime
from search_budget import over_budget

load_dotenv()

# 单独成文件是为了让"联网"这件事彻底可选：
# agent.py 只 import search_tools，没有 key 就拿到 None，整套本地功能照常跑。
# 如果直接在 agent.py 里 import langchain_tavily，没装这个包的人连本地版都启动不了。

_cache = None
# _loaded 和 _cache 是两个不同的东西，不能合并：
#   _loaded 记的是"已经查过了"，_cache 记的是"查到的结果"。
# 没配 key 时 _cache 永远是 None，只有靠 _loaded 才能保证那句提示只打印一次，
# 而不是每次调用都刷一遍屏。
_loaded = False


def search_tools():
    """返回 TavilySearch 实例；没配 TAVILY_API_KEY 时返回 None。"""
    global _cache, _loaded
    if _loaded:
        return _cache
    _loaded = True
    # 变量名必须全大写：langchain-tavily 只认这个名字，Linux 上也不区分大小写地说不通。
    # Windows 上小写也能跑（环境变量本身不区分大小写），所以在本地发现不了这个坑。
    if not os.getenv("TAVILY_API_KEY"):
        print("[tool_2] 未配置 TAVILY_API_KEY，联网搜索已关闭（本地检索不受影响）")
        return None
    # 延迟 import：不联网的时候连这个包都不用加载
    from langchain_tavily import TavilySearch
    # max_results=5 是权衡结果：再多会把上下文挤掉本地文献的位置
    _cache = TavilySearch(max_results=5, topic="general")
    return _cache


# 联网工具只是包一层：没配 TAVILY_API_KEY 时返回一句话让模型退回去用本地检索，
# 而不是让整个问答页报错。返回拼好的纯文本，模型只认文本。
#
# 开关必须在工具内部再核一遍。web_search 一直挂在工具列表里（见下面的 main_agent），
# switch 关掉只是换了提示词——模型看到的 schema 没变，遇到库里没有的问题照样会调。
# 本地 .env 里有 tavily_api_key，Windows 查环境变量不区分大小写，所以这台机器上是通的；
# 线上反而因为缺大写的 TAVILY_API_KEY 而调不动。结果是这个缺陷只在本机出现，
# 而本机正是录演示视频的地方：界面写着「仅本地文献库」，回答却来自互联网。
def web_search(query: str, runtime: ToolRuntime):
    """联网检索公开资料。只有本地文献库确实查不到相关内容时才使用。
        Args:
            query: 要在互联网上检索的关键词或问题
        """
    use_web = False
    if runtime.context:
        use_web = runtime.context.get("use_web", False)
    if not use_web:
        return ("联网搜索未开启（用户没有打开联网开关）。不要用你自己的知识回答，"
                "直接回答「库中没有相关资料」。")
    # 联网也占检索额度。不限制的话它一样能一轮搜二十几次（真发生过）。
    stop = over_budget(runtime)
    if stop:
        return stop
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
