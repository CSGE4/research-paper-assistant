"""
【前端交互限制与处理】
Streamlit采用重渲染机制，点击按钮后页面刷新，脚本会完整重跑，会二次进入文件上传分支。

处理方式：只在"已挂起、正等用户确认"时跳过重复解析（pending_task 里记着文件名）；
          其余情况都照常走——已入库的文件由 MD5 直接兜住（只比一次 md5，不解析、不嵌入），
          想重传一次也能重传。不要用"文件名+大小"之类的指纹去拦，那会把重传也拦掉。
"""
# 本文件是 Streamlit 的入口页（主页面），管三件事：
#   1. 上传 PDF 并入库
#   2. 上传时撞到相似文档，给出"覆盖/都留/取消"三个选择
#   3. 展示任意一篇已入库文档的结构化抽取结果
# 问答不在这个文件里，在 pages/app.py（Streamlit 会自动把 pages/ 下的文件变成侧边栏入口）。
#
# Streamlit 的执行模型要记住：每点一次按钮，整个脚本【从头重跑一遍】。
# 所以页面上的东西不是"事件驱动"的，而是"每次重跑时按当前状态重新画一遍"，
# 需要跨重跑保留的东西（服务对象、待确认任务）必须放进 st.session_state。

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

def clean_cell(value):
    """把 pymupdf4llm 残留的 markdown 标记去掉。

    它把数学式转成 markdown 时会写成 _O_(_n_<sup>2</sup>_·_d_) 这样，
    每个字符裹一层斜体标记、上标写成 <sup>。st.dataframe 不认这些，
    直接显示就是一堆下划线和尖括号。
    """
    s = str(value)
    s = s.replace("**", "")            # 粗体
    s = s.replace("_", "")             # 斜体标记
    s = re.sub(r"<sup>(.*?)</sup>", r"^(\1)", s)   # 上标 -> ^(2)
    s = re.sub(r"<sub>(.*?)</sub>", r"_(\1)", s)
    s = s.replace("</s>", "/s")        # 被上标截断的 /s 尾巴
    return s
def parse_agent_json(ai_msg_content: str):
    """
    从带```json标记的字符串提取json，增加异常捕获
    :param ai_msg_content: AIMessage.content 拿到的文本
    :return: dict / None 解析成功返回字典，失败返回None
    """
    # 智能体返回的是自然语言里夹一段 JSON，不能直接 json.loads。
    # 分两层兜底：先按 ```json 代码块取，取不到再按大括号取——
    # 模型偶尔会忘记写代码块标记，那时候只能靠大括号猜。
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
        # 模型输出中文时经常把 " 写成 “ ”，那是合法的中文标点但不是合法 JSON，
        # 不换掉的话 json.loads 一定失败。
        json_str = json_str.replace("“", '"').replace("”", '"')
        print(f"[parse_agent_json] 待解析字符串：{json_str}")

        data = json.loads(json_str)
        # 校验必须存在的三个key
        # 只校验字段在不在、不校验内容对不对：内容是给用户看的建议文案，
        # 缺字段才是程序会崩的地方（后面按 key 取值）。
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
    # 用全角空格按层级缩进，因为 Streamlit 的 markdown 会吃掉行首的普通空格，
    # 缩进就没了，层级看起来全是平的。
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
                # 表头和每个格子都先洗一遍，不然 _O_(_n_<sup>2</sup>_·_d_) 这种标记会原样显示。
                # 表头还可能有重名：论文里左右并排的两段表很常见（比如两组 Model/Race/Gender/Age），
                # pyarrow 把 DataFrame 转成 Arrow 时不允许重名列，会直接抛 ValueError，
                # 所以重名的后面补 _2、_3 区分，只影响显示。
                header = []
                seen = {}
                for i, h in enumerate(tb["header"] or []):
                    name = clean_cell(h).strip()
                    # 空表头 pandas 也会报错，补一个"列N"占位
                    if not name:
                        name = "列%d" % (i + 1)
                    if name in seen:
                        seen[name] = seen[name] + 1
                        name = "%s_%d" % (name, seen[name])
                    else:
                        seen[name] = 1
                    header.append(name)

                ncol = len(header)
                body = []
                # rows[0] 是表头，已经从 tb["header"] 拿过了，所以从 [1:] 开始
                for row in tb["rows"][1:]:
                    new_row = []
                    for cell in row:
                        new_row.append(clean_cell(cell))
                    # 格子数和表头对不上时补齐或截断，否则 pandas 也会报错
                    # 论文里的合并单元格经常导致某一行少几列，不能指望它对齐
                    while len(new_row) < ncol:
                        new_row.append("")
                    if len(new_row) > ncol:
                        new_row = new_row[:ncol]
                    body.append(new_row)

                if body and ncol > 0:
                    frame = pd.DataFrame(body, columns=header)
                elif body:
                    frame = pd.DataFrame(body)
                else:
                    # 只有表头没有数据行，也要画出来（这种表在论文里通常是纯表头示意）
                    frame = pd.DataFrame(columns=header)
                st.dataframe(frame, use_container_width=True, hide_index=True)
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
    # 留白会让人以为"这篇论文没有作者"，其实是没抽出来，性质完全不同。
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
                # 被丢弃的幻觉答案也展示出来，让人看到模型原本想写什么，
                # 这比只写"已丢弃"更能说明核对机制真的在工作
                if s == "rejected_hallucination":
                    extra = f"（模型曾给出：{det.get(f, {}).get('model')}）"
                elif s == "model" and det.get(f, {}).get("rule"):
                    extra = f"（规则原值：{det[f]['rule']}）"
                st.write(f"· {f}：{mark}{extra}")


def render_all_docs():
    """独立入口：列出已入库文档，选一个查看其结构。"""
    try:
        # 文档列表从结构化结果里取，不查向量库：
        # 两边理论上一致，但结构化文件才是这个页面真正要读的数据源
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

# service 和 pending_task 必须挂在 session_state 上：
# 页面每次重跑都会把模块级变量重置，只有 session_state 能跨重跑活下来。
# service 存的是 Knolege 实例（里面握着向量库连接和 BM25 索引，重建一次很贵）。
if "service" not in st.session_state:
    st.session_state["service"]=Knolege()
# pending_task 存"检测到相似文档，等用户按按钮"这件事。
# 它自己带着 pdf_base64，所以就算页面重跑、内存里的解析结果没了，
# 点按钮时也能靠它重新入库，不会因为缓存过期而崩。
if "pending_task" not in st.session_state:
    st.session_state.pending_task = None

if file is not None:
    name=file.name
    type=file.type
    size=file.size/1024
    st.write(f"文件名{name},文件类型{type}，文件大小{size:.2f}KB")
    # Streamlit 每次重跑都会带着上传组件里的文件再跑一遍脚本。
    # 只有"已经挂起、正等用户确认"这一种情况值得跳过：答案已经有了，
    # 再解析一遍纯属浪费一分钟，点按钮还会卡住。
    # 其余情况一律照常走：已入库的由 MD5 直接兜住（只比一次 md5，不解析不嵌入），
    # 想重传一次也能重传——不能用"文件名+大小"当指纹去拦，那会把重传也拦掉。
    pending=st.session_state.pending_task
    if pending is not None and pending.get("new_filename")==name:
        st.caption("这个文件正在等待确认，请看下方面板")
    else:
        with st.spinner("文件导入中"):
            file_type=type[-3:]
            if file_type=="pdf":
                a=file.read()
                # 走 base64 而不是字节流：后端接口统一收 base64 字符串，
                # 同一份内容 base64 编码结果稳定，md5 才能当作文件指纹用
                file_base_str=base64.b64encode(a).decode("utf-8")
                # 返回值有三种形态，见 knolege.upload_str_pdf 的说明：
                #   False → 完全重复；True → 已入库；dict → 有相似文档，等用户决定
                result,old_file_name,new_file_name=st.session_state["service"].upload_str_pdf(file_base_str,filename=name)
                if result is False:
                    st.warning("该文件MD5已存在，完全重复，跳过本次上传")
                    st.session_state.pending_task = None
                elif result is True:
                    st.success("文档入库完成，无相似文档")
                    st.session_state.pending_task = None
                else:
                    # 智能体的返回是一串消息，最后一条才是它写的内容
                    aimsg = result["messages"][1]
                    agent_raw_content = aimsg.content
                    agent_json = parse_agent_json(agent_raw_content)
                    # 组装 pending_task：这一步【没有入库】，只是把待确认的信息挂起来，
                    # 真正的写入发生在下面按钮分支里。
                    # pdf_base64 必须一起存：点按钮时可能已经过了几十秒，
                    # 内存里那份解析结果未必还在，到时候要靠它重新入库。
                    if agent_json is not None:
                        st.session_state.pending_task = {
                            "new_filename": new_file_name,
                            "old_filename": old_file_name,
                            "agent_suggest": f"""相似度判断：{agent_json["sim_desc"]}
                            推荐操作：{agent_json["action"]}
                            理由：{agent_json["reason"]}""",
                            "pdf_base64": file_base_str
                        }
                    else:
                        st.error("智能体返回结果解析失败，请重试上传！")
                        st.session_state.pending_task = None
            else:
                # txt / csv 目前只读不解析入库，属于预留入口
                text=file.getvalue().decode("utf-8")
# ===================== 【重点！！顶格写，没有缩进！！】=====================
# 这一段必须和上面的 if file is not None 平级。
# 原因：用户点"覆盖"按钮时页面重跑，这一次 file 是空的（上传组件被重置），
# 如果这段缩进在上面那个 if 里，按钮就永远不会被处理，点了没反应。
if st.session_state.pending_task is not None:
    task = st.session_state.pending_task
    st.divider()
    st.subheader("检测到相似文档，智能体给出分析建议")
    st.info(task["agent_suggest"])
    # 点按钮后可能要走一次重新解析（内存里的缓存过期时），最长约一分钟。
    # 提前说清楚，免得用户以为没反应又重传一遍，最后库里多出一份。
    st.caption("点按钮后最多需要约 1 分钟，请不要刷新页面或重复上传")

    col1, col2, col3 = st.columns(3)
    with col1:
        btn_overwrite = st.button("✅覆盖旧文档入库", type="primary")
    with col2:
        btn_both = st.button("📄两篇全部保留")
    with col3:
        btn_cancel = st.button("❌取消本次上传")

    if btn_overwrite:
        # 同名重传必须先删再写：块数可能变少，不删的话旧的高编号块会残留，
        # 变成"搜得到但内容已过期"的脏数据。
        # 不同名时反过来——先写新的、成功了再删旧的。
        # 顺序反了的话，写新的这一步一旦失败或超时，旧的已经被删掉，两边都没了。
        same_name=(task["old_filename"]==task["new_filename"])
        if same_name:
            st.session_state["service"].delete_file(task["old_filename"])
        # 带上 base64 一起传：内存里那份解析结果未必还在，
        # 没了就靠它重新解析入库，不再依赖"上传时的缓存还活着"这个前提
        saved,_,_=st.session_state["service"].real_save_pdf(task["new_filename"],task["pdf_base64"])
        if saved is not None and not same_name:
            st.session_state["service"].delete_file(task["old_filename"])
        st.session_state.pending_task = None
        if saved is None:
            st.error("待入库的解析结果已丢失，请重新上传这个文件")
        else:
            st.success("操作完成：已覆盖旧文档，新文档入库成功！")
            st.rerun()

    if btn_both:
        saved,_,_=st.session_state["service"].real_save_pdf(task["new_filename"],task["pdf_base64"])
        st.session_state.pending_task = None
        if saved is None:
            st.error("待入库的解析结果已丢失，请重新上传这个文件")
        else:
            st.success("操作完成：新旧文档全部入库！")
            st.rerun()

    if btn_cancel:
        # 取消只是把挂起的任务扔掉，什么数据都没写过，所以不需要回滚任何东西
        # 这里不 rerun：rerun 会把上面这句提示立刻冲掉，看起来像"点了没反应"
        st.warning("已取消，本次文件不会入库")
        st.session_state.pending_task = None

# ===================== 文档结构化结果查看（独立入口）=====================
st.divider()
st.header("文档结构化结果")
render_all_docs()
