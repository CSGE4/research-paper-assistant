"""
【已知前端交互限制】
Streamlit采用重渲染机制，点击确认按钮后页面刷新，脚本会完整重跑，触发二次进入文件上传分支。
防护机制：使用MD5校验文件指纹，第二次执行时直接拦截入库，不会向向量库、摘要库写入重复数据。
缺陷影响：会重复执行PDF解析、向量生成，增加接口耗时与token消耗。
未来优化方案：利用session_state增加文件处理标记，区分已处理任务，跳过重复后端计算。
"""
# Streamlit Cloud 的 Secrets 不会自动变成环境变量，
# 而这个项目里所有模块读的都是 os.getenv，所以第一件事是把 Secrets 灌进 os.environ。
# 必须放在下面这些 import 之前：vctor 在 import 的那一刻就会 new 一个 Knolege()。
import streamlit as st
import os
try:
    for k in st.secrets:
        os.environ[k] = str(st.secrets[k])
except Exception:
    # 本地跑的时候没有 secrets.toml，这里会抛异常，忽略即可 —— 本地读的是 .env
    pass
from knolege import *
import json
import re
import json
import re
import pandas as pd
from functools import lru_cache
from paper_extractor import DocumentParser, load_structured
from vctor import vctor_story_search


def parse_agent_json(ai_msg_content: str):
    """
    从带```json标记的字符串提取json，增加异常捕获
    :param ai_msg_content: AIMessage.content 拿到的文本
    :return: dict / None 解析成功返回字典，失败返回None
    """
    try:
        # 正则匹配 ```json 与 ``` 之间内容，re.S 让.匹配换行
        pattern = r"```json(.*?)```"
        match = re.search(pattern, ai_msg_content, re.S)
        if match:
            json_str = match.group(1).strip()
        else:
            # 兜底：如果模型忘记写```标记，捕获{}整体，加捕获括号 ()
            pattern2 = r"(\{.*?\})"
            match = re.search(pattern2, ai_msg_content, re.S)
            if not match:
                print("[parse_agent_json] 未找到JSON大括号")
                return None
            json_str = match.group(1).strip()

        # 清洗中文引号
        json_str = json_str.replace("“", '"').replace("”", '"')
        print(f"[parse_agent_json] 待解析字符串：{json_str}")

        data = json.loads(json_str)
        # 校验必须存在的三个key
        required_keys = {"sim_desc", "action", "reason"}
        if not required_keys.issubset(data.keys()):
            print(f"JSON缺少必要字段，当前keys:{data.keys()}")
            return None
        return data
    except Exception as e:
        # 任何错误：正则失败、json解析报错、字段缺失都返回None
        print(f"JSON解析异常: {e}")
        return None


def render_doc(name: str):
    """把某个文件的结构化提取结果画到页面上。

    数据来自入库时落盘的 chroma_structured.json（由 knolege 入库流程写入），
    因此页面不再重复解析 PDF，也不依赖向量库。
    """
    d = load_structured().get(name)
    if not d:
        st.info(f"「{name}」没有结构化结果（可能是本次改动之前入库的文件，重新上传一次即可）")
        return

    # 标题 + 作者
    tv = (d["title"] or {}).get("value")
    if tv:
        st.subheader(tv)
    au = (d["authors"] or {}).get("value") or []
    if au:
        st.caption("作者：" + "、".join(au))

    # 摘要 / 关键词
    ab = (d["abstract"] or {}).get("value")
    if ab:
        with st.expander("摘要", expanded=True):
            st.write(ab)
    kw = (d["keywords"] or {}).get("value") or []
    if kw:
        st.write("**关键词**：" + "、".join(kw))

    # 标题层级（level 1/2/3），大标题加粗
    st.markdown("#### 文档结构")
    for h in d["headings"]["tree"]:
        num = f"{h['number']} " if h["number"] else ""
        bold = "**" if h.get("kind") == "title" else ""
        st.markdown(f"{'　' * (h['level'] - 1)}{bold}{num}{h['title']}{bold}")

    # 表格：有行的直接喂 dataframe
    if d["tables"]:
        st.markdown(f"#### 表格（{len(d['tables'])}）")
        for tb in d["tables"]:
            cap = tb.get("caption") or "（无题注）"
            if tb.get("rows"):
                st.markdown(f"**{cap}**　{tb['n_rows']}行×{tb['n_cols']}列")
                st.dataframe(
                    pd.DataFrame(tb["rows"][1:], columns=tb["header"] or None),
                    use_container_width=True,
                )
            else:
                st.markdown(f"- {cap}（仅识别到题注，未取到表格数据）")

    # 公式
    if d["formulas"]:
        st.markdown(f"#### 公式（{len(d['formulas'])}）")
        for f in d["formulas"]:
            num = f"　({f['number']})" if f.get("number") else ""
            st.code(f["value"] + num, language=None)

    # 参考文献
    refs = d["references"]
    if refs["items"]:
        st.markdown(f"#### 参考文献（{refs['count']}）")
        for i, it in enumerate(refs["items"], 1):
            st.markdown(f"{i}. {it}")

    # 附录
    for ap in d["appendices"]:
        with st.expander(f"附录：{ap['title']}"):
            st.write(ap["content"])

    # 没识别到的字段明说，而不是静默留白
    if d["warnings"]:
        with st.expander(f"⚠️ 未能识别的字段（{len(d['warnings'])}）"):
            for w in d["warnings"]:
                st.write("· " + w)

    # 字段来源：标出每个值是规则给的还是模型补的，便于判断可信度
    src = d.get("extract_source") or {}
    det = d.get("field_details") or {}
    if src or d.get("model_used"):
        CN = {"rule": "规则解析", "model": "模型补全", "both": "规则+模型一致",
              "rejected_hallucination": "模型答案未通过核对，已丢弃"}
        with st.expander("🔎 字段来源（可追溯）"):
            if d.get("model_used"):
                st.caption(f"模型：{d['model_used']}")
            for f, s in src.items():
                if f == "model_error":
                    st.write(f"· 模型调用失败：{s}")
                    continue
                mark = CN.get(s, s)
                extra = ""
                if s == "rejected_hallucination":
                    extra = f"（模型曾给出：{det.get(f, {}).get('model')}）"
                elif s == "model" and det.get(f, {}).get("rule"):
                    extra = f"（规则原值：{det[f]['rule']}）"
                st.write(f"· {f}：{mark}{extra}")


def render_all_docs():
    """独立入口：列出已入库文档，选一个查看其结构。"""
    try:
        names = sorted(vctor_story_search().get_structured_all().keys())
    except Exception as e:
        st.warning(f"读取结构化结果失败：{e}")
        return
    if not names:
        st.info("还没有可展示的文档（本轮改动之后入库的文件才会有结构化结果）")
        return
    pick = st.selectbox("选择文档查看提取结果", names)
    if pick:
        render_doc(pick)


st.title("pdf文档上传")
file=st.file_uploader(
    "请上传文件",
    type=["pdf","txt","csv"],
    accept_multiple_files=False,
)

if "service" not in st.session_state:
    st.session_state["service"]=Knolege()
if "pending_task" not in st.session_state:
    st.session_state.pending_task = None

if file is not None:
    with st.spinner("文件导入中"):
        time1=datetime.datetime.now()
        name=file.name
        type=file.type
        size=file.size/1024
        st.write(f"文件名{name},文件类型{type}，文件大小{size:.2f}KB")
        file_type=type[-3:]
        if file_type=="pdf":
            a=file.read()
            file_base_str=base64.b64encode(a).decode("utf-8")
            result,old_file_name,new_file_name=st.session_state["service"].upload_str_pdf(file_base_str,filename=name)
            st.write(result)
            if result is False:
                st.warning("该文件MD5已存在，完全重复，跳过本次上传")
                st.session_state.pending_task = None
            elif result is True:
                st.success("文档入库完成，无相似文档")
                st.session_state.pending_task = None
            else:
                # 取出AIMessage
                aimsg = result["messages"][1]
                agent_raw_content = aimsg.content
                st.write("===调试原始内容===")
                st.code(agent_raw_content)

                agent_json = parse_agent_json(agent_raw_content)
                # 组装pending_task
                if agent_json is not None:
                    st.session_state.pending_task = {
                        "new_filename": new_file_name,
                        "old_filename": old_file_name,  # 这里你要从service里面拿到匹配到的旧文件名！
                        "agent_suggest": f"""相似度判断：{agent_json["sim_desc"]}
                        推荐操作：{agent_json["action"]}
                        理由：{agent_json["reason"]}""",
                        "pdf_base64": file_base_str
                    }
                else:
                    st.error("智能体返回结果解析失败，请重试上传！")
                    st.session_state.pending_task = None
        else:
            text=file.getvalue().decode("utf-8")
        time2=datetime.datetime.now()
        all_time=(time2-time1).total_seconds()
        st.write(all_time)
# ===================== 【重点！！顶格写，没有缩进！！】=====================
if st.session_state.pending_task is not None:
    task = st.session_state.pending_task
    st.divider()
    st.subheader("检测到相似文档，智能体给出分析建议")
    st.info(task["agent_suggest"])

    col1, col2, col3 = st.columns(3)
    with col1:
        btn_overwrite = st.button("✅覆盖旧文档入库", type="primary")
    with col2:
        btn_both = st.button("📄两篇全部保留")
    with col3:
        btn_cancel = st.button("❌取消本次上传")

    if btn_overwrite:
        st.session_state["service"].delete_file(task["old_filename"])
        st.session_state["service"].real_save_pdf(task["new_filename"])
        st.success("操作完成：已覆盖旧文档，新文档入库成功！")
        st.session_state.pending_task = None
        st.rerun()

    if btn_both:
        st.session_state["service"].real_save_pdf(task["new_filename"])
        st.success("操作完成：新旧文档全部入库！")
        st.session_state.pending_task = None
        st.rerun()

    if btn_cancel:
        st.warning("已取消，本次文件不会入库")
        st.session_state.pending_task = None
        st.rerun()

# ===================== 文档结构化结果查看（独立入口）=====================
st.divider()
st.header("文档结构化结果")
render_all_docs()