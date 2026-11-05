"""检索额度：一轮问答最多检索几次，以及怎么数。

单独成文件是因为 local_search（tool_3）和 web_search（tool_2）都要用它，
放在任何一个工具文件里都会让另一个反向依赖。
"""

# 一轮问答最多检索几次。这个数以前只写在提示词里，模型可以不听：
# 实测同一轮搜了 22 次、跑满 250 秒才吐答案。现在搬进代码，到数就拒。
MAX_RETRIEVAL = 6

# 哪些工具算「一次检索」。web_search 也占额度，否则联网开着时它照样能一直试。
SEARCH_TOOL_NAMES = ("local_search", "web_search")


def used_retrieval(runtime):
    """本轮已经完成的检索次数：从消息历史往前数到最近一条用户提问为止。

    不用模块级计数器：Streamlit 每次交互都会把脚本从头重跑，全局变量既会
    串到别的会话，也不会在下一轮自动归零（4.3 那几条缺陷就是这么来的）。
    历史里本来就记着每次调用，数它最准，也不需要额外维护状态。

    数 ToolMessage 而不是 AIMessage 的 tool_calls，因为工具执行这一刻
    自己那条结果还没生成，ToolMessage 的条数正好等于「已经完成的次数」。
    """
    messages = (runtime.state or {}).get("messages")
    n = 0
    for msg in reversed(messages or []):
        if getattr(msg, "type", "") == "human":
            break
        if getattr(msg, "type", "") == "tool" and getattr(msg, "name", "") in SEARCH_TOOL_NAMES:
            n = n + 1
    return n


def over_budget(runtime):
    """到上限就返回一句打发模型的话，没到返回 None。"""
    if used_retrieval(runtime) < MAX_RETRIEVAL:
        return None
    return ("本轮检索次数已达上限（%d 次），不能再调用检索工具。"
            "请直接用已经查到的内容作答；查不到的部分就说明没查到，不要换关键词再试。" % MAX_RETRIEVAL)
