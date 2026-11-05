"""文献问答页面（Streamlit 多页面应用的第二页）。

为什么单独放一个文件：
web.py 里是「上传 → 入库 → 看结构化结果」这一条流程，问答是完全独立的另一件事。
Streamlit 会自动在侧边栏生成页面入口
"""

import json
import time
import uuid

import streamlit as st
from langchain.messages import HumanMessage

# 检索次数的口径必须和 agent 那边一致，所以常量、工具名单、拒绝文本的判定
# 全部从 search_budget 取，不在这里另写一份。
from search_budget import MAX_RETRIEVAL, SEARCH_TOOL_NAMES, is_refusal


@st.cache_resource
def get_agent():
    """Streamlit 每点一次交互都会把脚本从头重跑一遍，
    用 cache_resource 保证智能体（连同它背后的 sqlite 连接、向量库客户端）
    只在第一次提问时创建一次，后面直接复用，不会越用越慢。

    智能体只有一个：联网那件事靠 invoke 时传的 context 换提示词，
    不另外建 agent，所以这里不带参数。
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

    # 联网开关，默认关。关着的时候提示词里没有联网那段规则；
    # 打开之后这同一个智能体会拿到 main_prompt + web_prompt（见 promot.py）
    use_web = st.toggle(
        "联网搜索",
        value=False,
        help="开启后先查本地文献库，本地查不到再查互联网，回答里会标出链接。",
    )
    if use_web:
        st.caption("本次回答：先本地，后联网")
    else:
        st.caption("本次回答：仅本地文献库")

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

    # 2. 交给智能体，它自己决定调 local_search、translate_paper 还是 web_search
    with st.chat_message("assistant"):
        cfg = {"configurable": {"thread_id": st.session_state["thread_id"]}}
        # 边生成边显示。一次问答实测 18 秒左右，等全部生成完再上屏，
        # 用户前 18 秒看到的是一个转圈，会以为卡死了。
        # stream_mode="messages" 会把模型吐的每一小段即时抛出来。
        # 必须按这个顺序建：灰字那行在回答上面
        # 多轮检索时模型每轮都先说一句「让我先查一下」，这些过渡语原来会被拼进
        # 最终回答里，用户看到的是「我需要先了解…让我逐篇检索…」再接一段正经答案。
        # 这里把它们拎到上面一行灰色小字里，只留最近几条。
        log_box = st.empty()
        placeholder = st.empty()
        # 这句提示必须跟着流程往下换。实测「作者是谁」十几秒就完，
        # 但「超参数是多少」这种要查表的问题，模型会换几次关键词重查，
        # 能跑到一分钟以上——全程只写「检索文献中…」的话，用户看到的就是卡死。
        placeholder.info("正在检索文献库…")
        t_start = time.time()
        answer = ""
        log = []           # 前面几轮的过渡语
        tag_args = ""      # 攒模型吐出来的工具参数（一串 JSON 碎片）
        pending = False    # 模型刚点完工具、还没进工具节点
        failed = None
        # 真正执行的检索次数。以前是数模型点了几次工具（见下面 tool_call_chunks 那段），
        # 被拒的尝试也算，于是界面显示 16 次、实际只查了 6 次。
        # 现在改成数工具返回：工具节点每返回一次结果才 +1，拒绝文本不算。
        n_search = 0

        try:
            for chunk, meta in get_agent().stream(
                {"messages": [HumanMessage(question)]},
                config=cfg,
                # 开关从这里传进 agent：pick_prompt 读它决定用哪套提示词
                context={"use_web": use_web},
                stream_mode="messages",
            ):
                node = meta.get("langgraph_node")
                # 工具节点在跑：检索在这里，rag.py 里那次内部生成也在这里。
                # 第一次收到就换文案——说明检索已经有结果，模型正在读资料。
                if node != "model":
                    if pending:
                        pending = False
                        # 这一轮的参数已经用掉了，清空，否则第二次检索的关键词
                        # 会和第一次拼成一串，JSON 解析必然失败
                        tag_args = ""
                        placeholder.info("已检索到资料，正在组织回答…")
                    # 只有真正查过的才计数。到上限后模型还会点几次工具，
                    # 那几次返回的是拒绝文本，不该算进「检索了几次」里。
                    tool_name = getattr(chunk, "name", "") or ""
                    if tool_name in SEARCH_TOOL_NAMES:
                        tool_text = chunk.content
                        if not isinstance(tool_text, str):
                            tool_text = str(tool_text or "")
                        if not is_refusal(tool_text):
                            n_search = n_search + 1
                    continue

                # 下面都是 model 节点。选工具那一次也在 model 节点里，
                # 但 content 为空、只带 tool_call_chunks，要单独处理。
                if getattr(chunk, "tool_call_chunks", None):
                    # 走到这里说明这一轮是在点工具，它刚才吐的正文只是过渡语。
                    # 挪到灰字区，主区域立刻补一句提示——
                    # 只清空 answer 变量是没用的，屏幕上的字还在；而下一段
                    # 碎片如果是个换行，就会渲染出一个孤零零的 ▌，看起来像白板。
                    if answer.strip():
                        log.append(answer.strip())
                        del log[:-4]          # 只留最近四条，别把版面撑长
                        log_box.caption(" · ".join(log))
                        answer = ""
                        placeholder.info("正在继续检索…")
                    # 参数是一个字一个字吐出来的 JSON 碎片，攒齐了才解析得动。
                    # 解析成功就把检索关键词显示出来，让用户知道在查什么。
                    for tc in chunk.tool_call_chunks:
                        tag_args = tag_args + (tc.get("args") or "")
                    if tag_args:
                        pending = True
                        try:
                            q = json.loads(tag_args).get("query")
                        except Exception:
                            q = None
                        if q:
                            placeholder.info("正在检索：%s" % q)
                    continue

                piece = chunk.content
                if not piece:
                    continue
                if not isinstance(piece, str):
                    piece = str(piece)
                answer = answer + piece
                # 只有真有内容才画。全是空白字符的话画出来就是一个孤零零的光标，
                # 用户会以为回答没了。
                if answer.strip():
                    placeholder.markdown(answer + "▌")

            # 兜底：万一流式的节点名或字段在当前版本里对不上，会一个字都收不到。
            # 只在「没抛异常但空手而归」时走这条路；异常不走，否则一次失败
            # 要把整条链再付一遍，耗时翻倍。
            if not answer:
                result = get_agent().invoke(
                    {"messages": [HumanMessage(question)]},
                    config=cfg,
                    context={"use_web": use_web},
                )
                answer = result["messages"][-1].content
                # 有的模型会把 content 返回成列表（分段内容），这里统一转成字符串
                if not isinstance(answer, str):
                    answer = str(answer)
        except Exception as e:
            # 原来这里没有 try/except：异常直接变成页面上一片红字，而且
            # 会话历史里只留下一个没人回答的提问。下一轮模型看到这个空档，
            # 就会自己编一个「我查过了但没拿到返回」的理由（真发生过）。
            failed = e
            if isinstance(e, NotImplementedError):
                # 中间件要结束这一轮时，如果模型一口气点了个以上的工具，它只能用
                # NotImplementedError 报出来。那不是真出错，就是撞上限了。
                answer = ("这一轮的检索次数到了上限（%d 次），已经停下来。"
                          "可以把问题问得更具体一点，或者分几次问。" % MAX_RETRIEVAL)
                placeholder.warning(answer)
            else:
                answer = "这一轮没有跑完（%s）。这不是查不到，是没跑完，请再问一次。" % type(e).__name__
                placeholder.error(answer)

        # 检索次数到上限时，中间件会用它自己的内部提示结束这一轮，那句话可能被
        # 当成最终回答刷到屏幕上。它有两种写法，都要认：
        #   结束语："'web_search' tool call limit reached: run limit exceeded (2/1 calls)"
        #   塞给模型的："Tool call limit exceeded. Do not call 'web_search' again."
        # 所以只匹配两者共有的 "tool call limit"。
        if answer and "tool call limit" in answer.lower():
            answer = ("这一轮的检索次数到了上限（%d 次），已经停下来。"
                      "可以把问题问得更具体一点，或者分几次问。" % MAX_RETRIEVAL)

        if failed is None:
            if answer.strip():
                placeholder.markdown(answer)
            else:
                # 一个字都没收到就是没答出来，别把空字符串刷上屏变成白板
                answer = "这一轮没有拿到回答，请再问一次。"
                placeholder.warning(answer)

        # 检索几次、花了多久。平时是给用户一个交代，出问题时是唯一能从界面上看到的线索。
        # 数的是真正执行的检索，不是模型点了几次工具——两者在撞上限时会差出十几次。
        st.caption("共检索 %d 次，用时 %.0f 秒" % (n_search, time.time() - t_start))

    # 3. 存进历史，页面重跑时还要画出来
    st.session_state["history"].append({"role": "assistant", "text": answer})
