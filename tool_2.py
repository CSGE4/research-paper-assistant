"""Tavily 联网搜索（可选工具）。缺 key 时不要影响主流程。"""
import os
from dotenv import load_dotenv
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
