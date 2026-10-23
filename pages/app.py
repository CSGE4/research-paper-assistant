"""文献问答页面（Streamlit 多页面应用的第二页）。

为什么单独放一个文件：
web.py 里是「上传 → 入库 → 看结构化结果」这一条流程，问答是完全独立的另一件事。
Streamlit 会自动在侧边栏生成页面入口
"""

import uuid

import streamlit as st
from langchain.messages import HumanMessage


@st.cache_resource
def get_agent():
    """Streamlit 每点一次交互都会把脚本从头重跑一遍，
    用 cache_resource 保证智能体（连同它背后的 sqlite 连接、向量库客户端）
    只在第一次提问时创建一次，后面直接复用，不会越用越慢。
    """
    from agent import main_agent
    return main_agent


st.title("文献问答")

# 每个浏览器会话一个 thread_id：checkpointer 是按 thread_id 存对话历史的，
# 大家共用一个固定 id 的话，别人打开页面会看到上一个人的对话
if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = str(uuid.uuid4())[:8]

# 页面上要画的对话记录。和智能体自己的记忆是两份：
# 这份只活在当前浏览器会话里，刷新页面不丢，关掉标签页就没了
if "history" not in st.session_state:
    st.session_state["history"] = []

with st.sidebar:
    st.caption("当前会话：" + st.session_state["thread_id"])
    if st.button("清空本次对话"):
        st.session_state["history"] = []
        # 换一个 thread_id 等于让智能体开一段新记忆，旧记忆还留在库里，不影响使用
        st.session_state["thread_id"] = str(uuid.uuid4())[:8]
        st.rerun()

# 先把历史记录画出来，再画输入框
for item in st.session_state["history"]:
    role = item["role"]
    text = item["text"]
    with st.chat_message(role):
        st.markdown(text)

question = st.chat_input("问论文里的内容，例如：Transformer 的多头注意力是怎么算的？")

if question:
    # 1. 用户的问题先上屏
    st.session_state["history"].append({"role": "user", "text": question})
    with st.chat_message("user"):
        st.markdown(question)

    # 2. 交给智能体，它自己决定调 local_search 还是 translate_paper
    with st.chat_message("assistant"):
        with st.spinner("检索文献中…"):
            cfg = {"configurable": {"thread_id": st.session_state["thread_id"]}}
            result = get_agent().invoke(
                {"messages": [HumanMessage(question)]},
                config=cfg,
            )
            answer = result["messages"][-1].content
            # 有的模型会把 content 返回成列表（分段内容），这里统一转成字符串
            if not isinstance(answer, str):
                answer = str(answer)
        st.markdown(answer)

    # 3. 存进历史，页面重跑时还要画出来
    st.session_state["history"].append({"role": "assistant", "text": answer})
