"""Tavily 联网搜索（可选工具）。缺 key 时不要影响主流程。"""
import os
from dotenv import load_dotenv
load_dotenv()

_cache = None
_loaded = False


def search_tools():
    """返回 TavilySearch 实例；没配 TAVILY_API_KEY 时返回 None。"""
    global _cache, _loaded
    if _loaded:
        return _cache
    _loaded = True
    if not os.getenv("TAVILY_API_KEY"):
        print("[tool_2] 未配置 TAVILY_API_KEY，联网搜索已关闭（本地检索不受影响）")
        return None
    from langchain_tavily import TavilySearch
    _cache = TavilySearch(max_results=5, topic="general")
    return _cache