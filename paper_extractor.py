# -*- coding: utf-8 -*-
"""
paper_extractor.py —— 从向量数据库的切块中提取文档结构化信息。

数据来源说明（与 knolege.py 保持一致）：
    入库时每一页 PDF 由 pymupdf4llm 转成 markdown，再用 SemanticChunker/
    RecursiveCharacterTextSplitter 切块，metadata 固定为：
        {"source": 文件名, "create_time": ..., "page_number": 页码}
    id 形如 "文件名_p页码_页内块序号"。

因此本模块的分工是：
    1) ChunkAssembler.process()  —— 把一个文件的所有碎片还原成「按页排好序的整篇 markdown」
    2) DocumentParser.parse()    —— 只做规则/正则解析，不调用大模型，纯本地可跑
    3) PaperExtractor.get_all()/get_designate(filename) —— 与 vctor_story_search 的接口对齐

提取字段：
    title 标题 | authors 作者 | abstract 摘要 | keywords 关键词
    headings 一级/二级/三级标题 | tables 表格 | figures 图 | formulas 公式
    references 参考文献 | appendices 附录 | sections 正文章节正文
"""

import datetime
import json
import os
import re

import config

__all__ = [
    "ChunkAssembler", "DocumentParser", "PaperExtractor",
    "extract_from_docs", "extract_from_pages", "PAPER_KEYS",
    "load_structured", "save_structured", "upsert_structured", "drop_structured",
]

# 结构化存储的版本号：以后字段变了可以据此重算
STRUCT_VERSION = 1

# 与解析结果一一对应的字段（顺序即报告展示顺序）
PAPER_KEYS = ("title", "authors", "abstract", "keywords", "headings",
              "tables", "figures", "formulas", "references", "appendices")

# ============================ 一、正则表 ============================

_MD_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*$")
_HRULE = re.compile(r"^-{3,}$|^\*{3,}$|^_{3,}$")
_BOLD_LINE = re.compile(r"^\*\*(.+?)\*\*[:：]?$")

# 裸标题：1 / 1. / 1.1 / 3.2.1 （要求后面确实是标题样文本，避免命中 "2023. 5. 1"）
_NUMBERED_HEADING = re.compile(
    r"^(\d{1,2}(?:\.\d{1,2}){0,2})\.?\s*[、.)）]?\s+(\S.*)$")

# 摘要 / 关键词 的行内标签
_ABSTRACT_MARK = re.compile(r"^\s*〔?(摘\s*要|Abstract|ABSTRACT)\s*[:：]?\s*(.*)$", re.I)
_KEYWORDS_MARK = re.compile(
    r"^\s*〔?(关\s*键\s*词|关键字|Key\s*words?|Keywords?)\s*[:：]\s*(.*)$", re.I)

_REFS_MARK = re.compile(r"^\s*〔?(参考文献|References?|BIBLIOGRAPHY|Bibliography)\s*[:：]?\s*$", re.I)
_APPENDIX_MARK = re.compile(
    r"^\s*(附\s*录\s*([A-Za-z0-9一二三四五六七八九十]+)|Appendix\s*([A-Za-z0-9]+))"
    r"\s*[:：、.]?\s*(.*)$", re.I)

# 字母编号的附录：有些论文不写 "Appendix" 这个词，直接写 "A. Linear-probe evaluation"。
# 这条规则不能单独用 —— 正文里也会出现 "A. xxx" 这种行，必须限定位置才算附录
# （参考文献之后，或者扫到 "Appendices / 附录" 大标题之后），
# 位置约束在 _classify_blocks 末尾单独处理。
_APPENDIX_LETTER = re.compile(r"^[A-Z](?:\.|\s+)\s*[A-Z0-9]")

# 只用来"开闸"的附录大标题：Appendices / Appendix / 附录，后面不带字母。
# 有的论文（Scaling Laws）把附录排在参考文献前面，光靠"参考文献之后"这个位置条件抓不到，
# 所以扫到这种大标题时也要放开判定。
_APPENDIX_GATE = re.compile(r"^\s*(Appendi(?:x|ces)|附\s*录)\s*[:：]?\s*$", re.I)

# 作者行：至少带一个分隔符，避免把标题行当作者
_AUTHOR_LABEL = re.compile(r"^\s*(作\s*者|Authors?|By)\s*[:：]\s*(.+)$", re.I)
_AUTHOR_SPLIT = re.compile(r"[,，;；、]|\s+and\s+|\s*&\s*")
_AUTHOR_BLOCKED = re.compile(
    r"大学|学院|学校|研究院|研究所|实验室|课题组|有限公司|通信作者|通讯作者|"
    r"博士|教授|硕士|导师|@|http|www\.|\d", re.I)
_CN_NAME = re.compile(r"^[\u4e00-\u9fa5]{2,4}$")
# 英文人名要用 Unicode 感知的写法：Łukasz / Çağlar / Gülçehre 这类名字
# 用 [A-Z] 开头会把它们全部漏掉。
# \u00B4 是 PDF 把 "ń/ź" 抽成 "´" 的情况（Koco´n / Wo´zniak），不加就整块被当成人名以外的东西。
_EN_NAME = re.compile(
    r"^[^\W\d_][\w.\-'\u00B4]*(?:\s+[^\W\d_][\w.\-'\u00B4]*){1,3}$", re.UNICODE)
# 姓可以带小写词缀：van Merrienboer / de la Cruz
_EN_NAME_LOOSE = re.compile(
    r"^[^\W\d_][\w.\-'\u00B4]*(?:\s+[^\W\d_][\w.\-'\u00B4]*)+$", re.UNICODE)
# 纯数字/符号片段：上标编号 "1,2" 会被 PDF 写成加粗，它不是人名也不是机构名，
# 遇到它不能当成"作者块结束"。
_ONLY_DIGITS = re.compile(r"^[\d\s,.;:%\-–—*_∗†‡§¶]+$")

# 图/表标题（中英）
_CAPTION_TABLE = re.compile(
    r"^\s*(?:表|Table|TABLE|Tab\.)\s*([A-Za-z]?\d{1,2}(?:[-.–]\d{1,2})?)"
    r"\s*[:：.、]?\s*(.*)$")
_CAPTION_FIGURE = re.compile(
    r"^\s*(?:图|Figure|FIGURE|Fig\.|FIG\.)\s*([A-Za-z]?\d{1,2}(?:[-.–]\d{1,2})?)"
    r"\s*[:：.、]?\s*(.*)$")

_MD_TABLE_SEP = re.compile(r"^\s*\|?[\s:\-|]+\|[\s:\-|]+$")
_FORMULA_INLINE = re.compile(r"\$([^$\n]{2,400})\$")
_FORMULA_BLOCK = re.compile(r"\$\$(.+?)\$\$|\\\[(.+?)\\\]", re.S)

_REF_ENTRY = re.compile(
    r"^\s*(?:\[\d{1,3}\]|〔\d{1,3}〕|\d{1,3}[.、])\s*\S")
# 章节编号（1. / 2.1）不能被当成参考文献条目
_SUBSECTION_NUM = re.compile(r"^\s*\d{1,2}(?:\.\d{1,2}){1,2}[.、]?\s")
# markdown 列表符号：pymupdf4llm 常把文献写成 "- [1] xxx"
_LIST_MARK = re.compile(r"^\s*[-*+]\s+")
_REF_CITATION = re.compile(r"\[\d{1,3}(?:\s*[-–,，]\s*\d{1,3})*\]")

_PICTURE_TEXT = re.compile(
    r"<!--\s*Start of picture text\s*-->(.*?)"
    r"(?=(?:<!--\s*End of picture text\s*-->\s*)?<!--\s*Start of picture text\s*-->"
    r"|<!--\s*End of picture text\s*-->|\Z)", re.S | re.I)
_PICTURE_TEXT_LOOSE = _PICTURE_TEXT          # 兼容旧名
_MD_IMAGE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")
_HTML_COMMENT = re.compile(r"<!--.*?-->", re.S)
_BR_TAG = re.compile(r"<br\s*/?>\s*", re.I)


def _clean_html_comments(s):
    """去掉 pymupdf4llm 的版式注释（<!-- ... -->），但保留同一行的正文/OCR 文字。"""
    return _HTML_COMMENT.sub("", s or "").strip()


def _norm_text(s):
    """行内 <br> 归一为空格（跨行 OCR 文字在同一个"行"里）。"""
    return _BR_TAG.sub(" ", s or "").strip()


def _has_html_comment(s):
    """整行只是版式标记（没有正文）时为 True。"""
    return bool(s) and bool(_HTML_COMMENT.search(s)) and not _clean_html_comments(s)
# 分页标记：本模块写入 <!-- page:N -->；同时兼容旧版 "{-N-}---" 写法
_PAGE_TAG = re.compile(r"<!--\s*page\s*:\s*-?\d+\s*-->|\{-?\d+-\}[\s\-.=*_]*")


# ============================ 二、小工具 ============================

def _lines(text):
    """保留原始换行结构地切行，同时统一换行符。"""
    return [ln.rstrip() for ln in (text or "").replace("\r\n", "\n").replace("\r", "\n").split("\n")]


def _clean_cell(cell):
    """表格单元格清洗：去 <br>、管道转义、多余空白。"""
    cell = (cell or "").replace("<br>", " ").replace("<br/>", " ").replace("\\|", "|")
    return re.sub(r"\s+", " ", cell).strip()


def _split_row(row):
    """把 '| a | b |' 切成 ['a', 'b']，容忍首尾缺省管道。"""
    row = row.strip()
    if row.startswith("|"):
        row = row[1:]
    if row.endswith("|"):
        row = row[:-1]
    return [_clean_cell(c) for c in row.split("|")]


def _cap_math(s, limit=400):
    s = (s or "").strip()
    return s if len(s) <= limit else s[:limit] + "…"


def _strip_md(s):
    """去掉行内 markdown 标记（粗体/斜体/代码/图片文字/<br>），但保留行首的 # 标题标记。"""
    s = re.sub(r"\*\*(.+?)\*\*", r"\1", s or "")
    s = re.sub(r"(?<!\*)\*([^*\n]+?)\*(?!\*)", r"\1", s)
    s = re.sub(r"`(.+?)`", r"\1", s)
    s = _PICTURE_TEXT.sub(lambda m: m.group(1), s)
    s = _BR_TAG.sub(" ", s)
    return s.strip()


def _strip_hashes(s):
    """只去掉行首的 markdown 标题井号，得到纯标题文本。"""
    return re.sub(r"^#{1,6}\s*", "", _strip_md(s)).strip()


def _strip_sup(s):
    """去掉 <sup>…</sup> / <sub>…</sub>，只给"作者名"这一路用。

    作者行里的上标是单位编号和脚注符号（Tri Dao<sup>†</sup>），不清掉的话
    "Tri Dao<sup>†</sup>" 既不像人名，又会把整行撑过长度上限。
    注意别改 _strip_md：公式和表格还要靠这些标签还原上下标。
    """
    s = re.sub(r"<sup>.*?</sup>", "", s or "", flags=re.I | re.S)
    s = re.sub(r"<sub>.*?</sub>", "", s, flags=re.I | re.S)
    return s


# 一级标题里属于论文固定结构的名字（这些不该被当成文章标题）
_STANDARD_SECTION = re.compile(
    r"^\s*(?:\d{1,2}(?:\.\d{1,2})*\.?\s*)?"
    r"(摘\s*要|Abstract|ABSTRACT|引\s*言|绪\s*论|前\s*言|结\s*论|结\s*果|讨\s*论|"
    r"参考文献|References?|Bibliography|致\s*谢|Acknowledg\w*|"
    r"关\s*键\s*词|Keywords?|附\s*录|Appendix)", re.I)


def _numbers_in(text):
    return [float(x) for x in re.findall(r"\d+(?:\.\d+)?", text or "")]


def _norm_number(num, text):
    """把 '1.1' 归一成 [1, 1]；若标题里写 '一、' 则用中文数字兜底。"""
    if num:
        return [int(p) for p in str(num).split(".") if p.isdigit()]
    m = re.match(r"^\s*([一二三四五六七八九十]+)\s*[、.．]", _strip_md(text))
    if m:
        return [{"一": 1, "二": 2, "三": 3, "四": 4, "五": 5,
                 "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}.get(m.group(1), 1)]
    return []


def _split_name_run(text):
    """把挤在同一个加粗块里的多个名字拆开。

    有的 PDF 只把前两个作者单独加粗，剩下的并成一块：
        **Phillip Wallis Zeyuan Allen-Zhu Yuanzhi Li Shean Wang Lu Wang Weizhu Chen**
    一个名字一般两三个词，所以每次取两个词；如果取到的以 "." 结尾（中间名缩写
    如 "Tom B."），再并上后面的词当完整的姓。

    拆完逐个用 _EN_NAME 验一遍，只要有一个不像人名，就整块返回空、让调用方放弃，
    宁可少几个作者也不硬猜（免得把机构名、致谢里的话拼进作者列表）。
    """
    words = (text or "").split()
    if len(words) < 4:
        return []
    groups = []
    i = 0
    while i < len(words):
        if i + 1 < len(words):
            pair = words[i] + " " + words[i + 1]
        else:
            pair = words[i]
        i = i + 2
        if pair.endswith(".") and i < len(words):
            pair = pair + " " + words[i]
            i = i + 1
        groups.append(pair)
    for g in groups:
        if not _EN_NAME.match(g):
            return []
    return groups


def _is_headish(text, allow_short=False):
    """判断一行去掉编号后是否"像标题"：短、无句末标点、不像整段正文。"""
    t = _strip_md(text)
    if not t:
        return False
    core = re.sub(r"^[\d.\s、)）]+", "", t).strip()
    core = re.sub(r"^(第[一二三四五六七八九十百\d]+[章节部分篇])\s*", "", core)
    if not core:
        return False
    if len(core) > 40:
        return False
    if len(core) < 2 and not allow_short:
        return False
    if re.search(r"[。；;！!？?]$", core):
        return False
    # 以逗号结尾、词数很多的，多半是正文
    if len(core) > 24 and re.search(r"[，,、]$", core):
        return False
    return True


def _numbered_parts(line):
    """返回 (编号列表, 标题文本) 或 None。"""
    m = _NUMBERED_HEADING.match(_strip_md(line))
    if not m:
        return None
    num, rest = m.group(1), m.group(2)
    if not _is_headish(rest):
        return None
    return _norm_number(num, line), rest


def _major_heading(line):
    """识别"大标题"（无编号）：摘要/参考文献/致谢/附录X 等。返回 (kind, 文本)。"""
    t = _strip_md(line)
    if not t or len(t) > 40:
        return None
    if _REFS_MARK.match(t):
        return "refs", t
    if _APPENDIX_MARK.match(t):
        return "appendix", t
    if re.match(r"^(摘\s*要|Abstract|ABSTRACT|引\s*言|绪\s*论|前\s*言|结\s*论|"
                r"结\s*果|讨\s*论|致\s*谢|Acknowledg)", t, re.I):
        return "heading", t
    return None


# ============================ 三、碎片装配 ============================

class ChunkAssembler:
    """把 vctor 拿到的切块还原成"按文件、按页排序"的整篇 markdown。"""

    def __init__(self, docs):
        self.docs = docs or {}
        self._groups = None

    # ---- 排序 ----
    @staticmethod
    def _chunk_index(cid, fallback):
        """从 id 尾部取页内序号：'四级.pdf_p1_3' -> 3。"""
        try:
            tail = str(cid).rsplit("_", 1)[-1]
            return int(tail)
        except (ValueError, AttributeError, IndexError):
            return fallback

    @staticmethod
    def _page_of(cid, meta, fallback):
        if meta and meta.get("page_number") is not None:
            return meta.get("page_number")
        m = re.search(r"_p(-?\d+)_", str(cid))
        return int(m.group(1)) if m else fallback

    def _build(self):
        if self._groups is not None:
            return self._groups
        ids = self.docs.get("ids") or []
        texts = self.docs.get("documents") or []
        metas = self.docs.get("metadatas") or []
        groups = {}
        for i in range(len(ids)):
            cid = ids[i]
            meta = metas[i] if i < len(metas) else None
            meta = meta or {}
            text = texts[i] if i < len(texts) else ""
            source = meta.get("source")
            if not source:
                m = re.match(r"^(.*?)_p-?\d+_\d+$", str(cid))
                source = m.group(1) if m else str(cid)
            g = groups.setdefault(source, {"chunks": [], "metas": [], "pages": []})
            g["chunks"].append({
                "index": i,
                "id": cid,
                "text": text or "",
                "metadata": meta,
                "page": self._page_of(cid, meta, 1),
                "chunk_index": self._chunk_index(cid, i),
            })
            g["metas"].append(meta)
            g["pages"].append(g["chunks"][-1]["page"])
        for source, g in groups.items():
            g["chunks"].sort(key=lambda c: (c["page"], c["chunk_index"], c["index"]))
        self._groups = groups
        return groups

    def sources(self):
        return sorted(self._build().keys())

    def assemble(self, source, mark_page=True):
        """返回整篇文档 markdown（块之间用空行拼接，可按页插入 {-页码-} 标记）。"""
        g = self._build().get(source)
        if not g:
            return ""
        out = []
        for c in g["chunks"]:
            if mark_page:
                out.append("<!-- page:%s -->" % c["page"])
            if c["text"]:
                out.append(c["text"].strip("\n"))
        return "\n\n".join(out)

    def build(self, source):
        """返回完整结构：文本、页范围、逐页文本、块列表。"""
        g = self._build().get(source)
        if not g:
            return None
        pages = g["pages"] or [None]
        per_page = {}
        for c in g["chunks"]:
            per_page.setdefault(c["page"], []).append(c["text"])
        return {
            "source": source,
            "text": self.assemble(source),
            "page_start": min(pages),
            "page_end": max(pages),
            "page_count": len(set(pages)),
            "pages": {p: "\n".join(t) for p, t in sorted(per_page.items())},
            "chunks": g["chunks"],
            "metadatas": g["metas"],
        }

    def all(self):
        """返回 {文件名: build(source)}。"""
        return {s: self.build(s) for s in self.sources()}


# ============================ 四、结构解析 ============================

class DocumentParser:
    """对还原后的整篇 markdown 做规则解析，得到结构化信息。"""

    def __init__(self, config=None):
        c = config or {}
        self.title_max_len = c.get("title_max_len", 200)
        self.caption_window = c.get("caption_window", 3)
        self.keep_empty = c.get("keep_empty", True)

    # ---- 主入口 ----
    def parse(self, doc, include_sections=True):
        text = doc["text"]
        # 隐藏页码标记，但保持行数不变，这样 zip(...) 不会截断文档尾部
        prepped = _PAGE_TAG.sub("", text)
        lines = _lines(prepped)

        blocks = [{"line": ln, "page": pg} for ln, pg in zip(lines, self._page_map(text))]
        # 标题先算：后面要拿它把"大标题那一行"排除出小节
        title = self._title(doc, [], text)
        self._tag_headings(blocks, (title or {}).get("value"))
        structure = self._split_sections(blocks)
        result = self._fields(doc, prepped, structure, title=title)
        if include_sections:
            result["sections"] = self._section_items(structure)
        return result

    @staticmethod
    def _page_map(text):
        """逐行标注页码（靠 assemble() 写入的 {-n-} 标记）。
        必须与 _lines() 行数严格一致，否则 zip 会静默截断文档尾部。"""
        pages = []
        cur = None
        for ln in _lines(text):
            m = re.match(r"^\s*<!--\s*page\s*:\s*(-?\d+)\s*-->", ln) or \
                re.match(r"^\s*\{-(\d+)-\}", ln)
            if m:
                cur = int(m.group(1))
            pages.append(cur)
        return pages

    # ---- 标题识别 ----
    def _tag_headings(self, blocks, title_value=None):
        for b in blocks:
            b["type"], b["level"], b["num"], b["title"] = "text", None, [], None
            raw = b["line"]
            # 分页标记不是正文内容
            if not raw.strip() or _PAGE_TAG.match(raw.strip()) or _HRULE.match(raw.strip()):
                b["type"] = "blank"
                continue

            stripped = _strip_md(raw)
            # 文章大标题本身不是小节：这里要用去掉井号后的纯文本比较，
            # 否则 "# 标题"（stripped 带 #）匹配不上 title 值，标题会被当成普通小节，
            # 也不会出现在 _headings 里。
            if title_value and title_value in (stripped, _strip_hashes(stripped)):
                continue

            # 1) markdown 标题
            m = _MD_HEADING.match(stripped)
            if m:
                title = _strip_hashes(m.group(2))
                lvl = min(len(m.group(1)), 3)
                num, kind = [], "heading"
                if _REFS_MARK.match(title):
                    kind, lvl = "refs", 1
                elif _APPENDIX_MARK.match(title):
                    kind, lvl = "appendix", 1
                else:
                    mp = _NUMBERED_HEADING.match(title)
                    if mp and _is_headish(mp.group(2)):
                        num, title = _norm_number(mp.group(1), title), mp.group(2)
                b.update(type="heading", level=self._norm_level(lvl, num, title),
                         num=num, title=title, kind=kind)
                continue

            # 2) 加粗整行
            mb = _BOLD_LINE.match(stripped)
            if mb and _is_headish(mb.group(1)):
                title = _strip_md(mb.group(1))
                kind = "refs" if _REFS_MARK.match(title) else (
                    "appendix" if _APPENDIX_MARK.match(title) else "heading")
                b.update(type="heading", level=1, num=[], title=title, kind=kind)
                continue

            # 3) 摘要/关键词行：标签后面还有正文时，这是正文而不是标题
            #    （PDF 直接转出来常见 "摘要：xxx"，没有 markdown 标记）
            #    注意：单纯的 "摘要" / "## 摘要" 不在这里拦，它仍是一个小节标题。
            abs_inline = _ABSTRACT_MARK.match(stripped)
            if abs_inline and (abs_inline.group(2) or "").strip():
                continue
            kw_inline = _KEYWORDS_MARK.match(stripped)
            if kw_inline and (kw_inline.group(2) or "").strip():
                continue

            # 4) 编号标题（无 markdown 标记）
            parts = _numbered_parts(stripped)
            if parts:
                num, title = parts
                b.update(type="heading", level=min(len(num), 3) or 1, num=num,
                         title=title, kind="heading")
                continue

            # 5) 大标题（无编号）
            major = _major_heading(stripped)
            if major:
                kind, title = major
                b.update(type="heading", level=1, num=[],
                         title=title if kind != "appendix"
                         else _APPENDIX_MARK.match(title).group(0).strip(),
                         kind=kind)
                continue

        # 补一遍字母编号的附录（"A. Linear-probe evaluation"）。
        # 必须限定位置：正文里也会出现 "A. xxx" 这种行，
        # 不加位置约束就会把正文章节误判成附录。
        # 两种放行条件：扫到了参考文献，或者扫到了 "Appendices" 这种附录大标题
        # （Scaling Laws 的附录排在参考文献前面，只有前者抓不到）。
        passed_refs = False
        passed_gate = False
        for b in blocks:
            if b["type"] != "heading":
                continue
            if b.get("kind") == "refs":
                passed_refs = True
                continue
            title = b.get("title") or ""
            if _APPENDIX_GATE.match(title):
                # 这行只是"Appendices / 附录"大标题，本身不算一条附录
                passed_gate = True
                continue
            if not (passed_refs or passed_gate):
                continue
            if b.get("kind") != "heading":
                continue
            if _APPENDIX_LETTER.match(title):
                b["kind"] = "appendix"

    # ---- 分节 ----
    @staticmethod
    def _norm_level(md_level, num, title):
        """让"一级/二级/三级标题"与编号对齐：
        1 -> 一级；1.1 -> 二级；1.1.1 -> 三级；带编号的标题不看 markdown 井号数量。
        没有编号时，文章大标题（H1）保持一级，其余按 markdown 级别。"""
        if num:
            return min(len(num), 3) or 1
        if md_level == 1:
            return 1
        # 摘要/致谢/关键词等固定章节没有编号，也归为一级标题
        if re.match(r"^\s*(摘\s*要|Abstract|ABSTRACT|致\s*谢|Acknowledg\w*|"
                    r"关\s*键\s*词|Keywords?)\s*$", title or "", re.I):
            return 1
        return min(md_level, 3)

    def _split_sections(self, blocks):
        sections = []
        current = None
        for b in blocks:
            if b["type"] == "heading":
                current = {
                    "title": b["title"], "level": b["level"], "num": b["num"],
                    "kind": b.get("kind", "heading"), "page": b["page"],
                    "lines": [], "parent": None,
                }
                sections.append(current)
                continue
            if current is None:
                continue
            current["lines"].append((b["line"], b["page"], b["type"]))
        self._link_parents(sections)
        return sections

    @staticmethod
    def _link_parents(sections):
        stack = []
        for sec in sections:
            key = sec["num"] or []
            while stack and (not stack[-1]["num"] or
                             (_numbers_in(".".join(map(str, stack[-1]["num"]))) >=
                              _numbers_in(".".join(map(str, key))))):
                stack.pop()
            sec["parent"] = stack[-1]["title"] if stack else None
            if key:
                stack.append(sec)
            else:
                stack = []

    @staticmethod
    def _section_items(sections):
        """把 sections 转成对外结构：path 为层级路径，content 为该节正文。"""
        items = []
        for sec in sections:
            content = "\n".join(ln for ln, _, _ in sec["lines"]).strip()
            items.append({
                "title": sec["title"],
                "level": sec["level"],
                "number": ".".join(str(x) for x in sec["num"]) or None,
                "parent": sec["parent"],
                "path": [p for p in [sec["parent"], sec["title"]] if p],
                "page": sec["page"],
                "kind": sec["kind"],
                "content": content,
                "length": len(content),
            })
        return items

    # ---- 各字段 ----
    def _fields(self, doc, text, sections, title=None):
        if title is None:
            title = self._title(doc, sections, text)
        authors = self._authors(text, title)
        abstract = self._abstract(text, sections)
        keywords = self._keywords(text)
        refs = self._references(text, sections)
        appendices = self._appendices(sections)
        tables = self._tables(text, sections)
        figures = self._figures(text, sections)
        formulas = self._formulas(text)
        headings = self._headings(sections)
        # 文章大标题也是结构的一部分：它被排除出小节后，这里补回结构的根
        title_val = (title or {}).get("value")
        if title_val and not any(h.get("title") == title_val for h in headings["tree"]):
            headings["tree"].insert(0, {
                "title": title_val, "level": 1, "number": None,
                "parent": None, "kind": "title",
                "page": 1 if doc.get("pages") else None,
            })
            headings["level1"].insert(0, headings["tree"][0])
        return {
            "source": doc["source"],
            "page_start": doc.get("page_start"),
            "page_end": doc.get("page_end"),
            "page_count": doc.get("page_count"),
            "chunk_count": len(doc.get("chunks") or []),
            "title": title,
            "authors": authors,
            "abstract": abstract,
            "keywords": keywords,
            "headings": headings,
            "tables": tables,
            "figures": figures,
            "formulas": formulas,
            "references": {
                "title": refs["title"],
                "count": len(refs["items"]),
                "items": refs["items"],
                "in_text_citations": refs["citations"],
            },
            "appendices": appendices,
            "warnings": self._warnings(doc, sections, title, authors, abstract, tables,
                                       figures, formulas, refs["items"], appendices),
        }

    def _title(self, doc, sections, text):
        source = doc["source"]
        stem = re.sub(r"\.(pdf|docx?|md|txt)$", "", source, flags=re.I)
        pages = doc.get("pages") or {}
        page1 = pages.get(min(pages)) if pages else None
        first_page = _lines(page1 if page1 is not None else text)

        # 1) 优先第一页的大标题（论文大标题常被 pymupdf4llm 输出成 "# 标题"）
        #    一级二级都算：有的论文（Scaling Laws）大标题是 "## 标题"，
        #    只认 "#" 的话会滑到下一行的加粗作者上。
        for ln in first_page:
            m = _MD_HEADING.match(_strip_md(ln))
            if not m or len(m.group(1)) > 2:
                continue
            t = _strip_hashes(m.group(2))
            if not t or _STANDARD_SECTION.match(t):
                continue
            # 带编号的一级标题（"1 Introduction"）是正文章节，不是文章大标题，
            # 否则它会抢在"加粗大标题"前面被当成标题（Attention 论文就是这样）。
            if _NUMBERED_HEADING.match(t):
                continue
            return self._title_result(t, stem)

        # 1b) 整行加粗的写法：**Attention Is All You Need**
        #     注意必须用"原文行"匹配：_strip_md 会把 ** 删掉，用 stripped 永远匹配不上。
        for ln in first_page:
            raw_line = _clean_html_comments(ln).strip()
            mb = _BOLD_LINE.match(raw_line)
            if not mb:
                continue
            if len(re.findall(r"\*\*(.+?)\*\*", raw_line)) != 1:
                continue                # 多个加粗片段（作者行）不算标题
            t = _strip_md(mb.group(1))
            if not t or len(t) < 4 or _has_html_comment(t):
                continue
            if _STANDARD_SECTION.match(t) and len(t) <= 12:
                continue                # 纯章节名（如 "Abstract"）留给后面的兜底
            return self._title_result(t, stem)

        # 2) 退化为第一页首个像正文的行（把版式注释清掉，保留同行 OCR 文字）
        candidates = []
        for ln in first_page:
            t = _strip_md(_clean_html_comments(ln))
            if not t or _MD_HEADING.match(t) or _HRULE.match(t):
                continue
            if len(t) < 4:
                continue
            if _ABSTRACT_MARK.match(t) or _KEYWORDS_MARK.match(t):
                continue
            if _is_headish(t, allow_short=True) or 4 <= len(t) <= self.title_max_len:
                candidates.append(t)
            if len(candidates) >= 3:
                break
        best = None
        for c in candidates:
            if len(c) <= self.title_max_len and not _AUTHOR_BLOCKED.search(c):
                best = c
                break
        if not best and candidates:
            best = candidates[0]
        # 3) 扫描件/证书：正文写在图片文字层里，取该层第一行的第一个片段
        if not best:
            for m in _PICTURE_TEXT.finditer("\n".join(first_page)):
                first = (m.group(1) or "").strip()
                if not first:
                    continue
                first = _strip_md(_clean_html_comments(first)).split("  ")[0].strip()
                if len(first) >= 4:
                    best = first
                    candidates.append(first)
                break
        return self._title_result(best, stem, candidates[:5])

    @staticmethod
    def _title_result(value, stem, candidates=None):
        return {
            "value": value,
            "guess_from_filename": stem,
            "matched_filename": bool(value and (value in stem or stem in value)),
            "candidates": candidates if candidates is not None else ([value] if value else []),
        }

    def _authors(self, text, title):
        lines = _lines(text)
        # 1) "作者：xxx" 显式标签
        for ln in lines[:80]:
            m = _AUTHOR_LABEL.match(_strip_md(ln))
            if m:
                names = self._split_authors(m.group(2))
                if names:
                    return {"value": names, "raw": m.group(2), "origin": "label"}
        # 2) 标题行后面紧跟的短行（带分隔符、像人名）
        title_val = (title or {}).get("value")
        start = 0
        if title_val:
            for i, ln in enumerate(lines[:80]):
                if title_val in _strip_md(ln):
                    start = i + 1
                    break
        plain_names, plain_raw = [], []
        for ln in lines[start:start + 6]:
            if _has_html_comment(ln):
                continue
            # 先把上标（单位编号/脚注符号）清掉再判断：不清的话
            # "Tri Dao<sup>†</sup> , Daniel Y. Fu..." 会既不像人名又超长被跳过。
            t = _strip_sup(_strip_md(ln))
            if not t or _has_html_comment(t):
                continue
            if _ABSTRACT_MARK.match(t) or _KEYWORDS_MARK.match(t):
                continue
            if _MD_HEADING.match(t) or _major_heading(t) or _STANDARD_SECTION.match(t):
                continue
            if _numbered_parts(t) or _APPENDIX_MARK.match(t):
                continue
            if len(t) > 120:
                continue
            names = self._split_authors(t)
            if len(names) >= 2:
                plain_names.extend(names)
                plain_raw.append(t[:100])
        # 2b) 标题之后连续若干行里的加粗作者块：
        #     **Ashish Vaswani**<sup>_∗_</sup> **Noam Shazeer**<sup>_∗_</sup> Google Brain ...
        #     学术 PDF 极常见。要"收集所有行"，不能遇到第一行就返回，
        #     否则后面的作者会丢（Attention 论文的作者分 3 行）。
        bold_names, bold_raw = [], []
        for ln in lines[start:start + 20]:
            found = self._names_from_bold_line(ln)
            if found:
                bold_names.extend(found)
                bold_raw.append(_strip_md(ln)[:100])
        # 加粗结果通常最可靠（作者名被显式强调），优先用它
        if bold_names:
            return {"value": bold_names, "raw": " / ".join(bold_raw)[:200],
                    "origin": "bold_spans"}
        if plain_names:
            return {"value": plain_names, "raw": " / ".join(plain_raw)[:200],
                    "origin": "line_after_title"}
        # 3) 扫描件/证书：从图片文字层里按 "名：/姓名：/作者：" 取
        pic = self._authors_from_picture_text(text)
        if pic:
            return pic
        return {"value": [], "raw": None, "origin": None}

    @staticmethod
    def _names_from_bold_line(line):
        """从"一行里多个加粗片段"中取出作者名。

        形如 **Ashish Vaswani**<sup>_∗_</sup> **Noam Shazeer**<sup>_∗_</sup> Google Brain ...
        做法：按顺序看加粗片段，遇到机构等非人名就停止（作者通常在最前面连续出现）。
        两种情况要特殊处理，否则只拿到前一两个作者：
          1) 上标编号本身也是加粗的（<sup>**1,2**</sup>），它不是人也不是机构，跳过别停；
          2) 多个名字被并进同一个加粗块，交给 _split_name_run 拆，拆不干净就整块不要。
        """
        raw = line or ""
        names = []
        for b in re.findall(r"\*\*(.+?)\*\*", raw):
            t = _strip_md(b)                       # 去掉 <sup>/<sub> 等
            if not t:
                continue                           # 空片段（如 <sup>_∗_</sup>）跳过
            if _EN_NAME.match(t) or _CN_NAME.match(t):
                names.append(t)
                continue
            if _ONLY_DIGITS.match(t):
                continue                           # 上标编号 "1,2"，跳过，不算作者块结束
            if len(t.split()) >= 4:
                parts = _split_name_run(t)         # 一块里挤了好几个名字
                if parts:
                    names.extend(parts)
                    continue
            if names:
                break                              # 已经收到人名，遇到机构等非人名就结束
        return names

    @staticmethod
    def _authors_from_picture_text(text):
        """扫描件没有版式信息，姓名往往挂在 OCR 图片文字层的 '名：' 字段上。"""
        for m in _PICTURE_TEXT.finditer(text or ""):
            block = _norm_text(_clean_html_comments(m.group(1) or ""))
            if not block:
                continue
            for mm in re.finditer(r"(?:姓\s*名|名|作\s*者)\s*[:：]\s*([^\s:：]{2,20})", block):
                cand = mm.group(1).strip(" 。，,、;；")
                if _CN_NAME.match(cand) or _EN_NAME.match(cand):
                    return {"value": [cand], "raw": block[:120], "origin": "picture_text"}
        return None

    @staticmethod
    def _split_authors(line):
        # 上标（单位编号/†‡ 脚注）先清掉，否则名字后面挂一堆标签，个个都不像人名
        raw = _strip_sup(_strip_md(line))
        if not raw or len(raw) > 160:
            return []
        parts = [p.strip(" *†‡§¶0123456789") for p in _AUTHOR_SPLIT.split(raw) if p.strip()]
        names = []
        for p in parts:
            p = p.strip()
            if not p or len(p) > 30 or _AUTHOR_BLOCKED.search(p):
                continue
            if _CN_NAME.match(p) or _EN_NAME.match(p):
                names.append(p)
        return names

    def _abstract(self, text, sections):
        # 1) 显式 "摘要/Abstract" 标记（保留行号，便于顺行向下取正文）
        raw_lines = _lines(text)
        for i, ln in enumerate(raw_lines):
            m = _ABSTRACT_MARK.match(_strip_md(ln))
            if not m:
                continue
            inline = (m.group(2) or "").strip()
            buf = [inline] if inline else []
            for nxt in raw_lines[i + 1:i + 60]:
                t = _strip_md(nxt).strip()
                if _PAGE_TAG.match(t):
                    continue
                if not t:
                    if buf and buf[-1] != "":
                        buf.append("")
                    continue
                if _KEYWORDS_MARK.match(t) or _MD_HEADING.match(t) or \
                        _major_heading(t) or _numbered_parts(t) or _REFS_MARK.match(t):
                    break
                buf.append(t)
                if len("".join(buf)) > 4000:
                    break
            body = "\n".join(buf).strip()
            if body:
                return {"value": body, "origin": "marker", "page": None}
        # 2) 退化为 "摘要" 小节内容
        sec = next((s for s in sections if re.match(r"^(摘\s*要|Abstract)", s["title"] or "", re.I)), None)
        if sec and sec["lines"]:
            body = "\n".join(ln for ln, _, _ in sec["lines"]).strip()
            if body:
                return {"value": body, "origin": "section", "page": sec["page"]}
        # 3) 没有任何标记（如扫描件/证书）：正文首段兜底，并标注来源
        for s in sections:
            if not s["num"] and s["lines"]:
                body = "\n".join(ln for ln, _, _ in s["lines"]).strip()
                if len(body) >= 60:
                    return {"value": body[:2000], "origin": "first_block_fallback", "page": s["page"]}
        return {"value": None, "origin": None, "page": None}

    def _keywords(self, text):
        for ln in _lines(_strip_md(text)):
            m = _KEYWORDS_MARK.match(ln)
            if not m:
                continue
            body = (m.group(2) or "").strip()
            if not body:
                continue
            items = [k.strip(" .。;；") for k in re.split(r"[,，;；、]\s*", body) if k.strip()]
            return {"value": items or [body], "raw": body}
        # 有些排版把关键词并进摘要同一行
        abs_val = self._abstract(text, [])["value"]
        if abs_val:
            m = re.search(r"(?:关\s*键\s*词|Key\s*words?|Keywords?)\s*[:：]\s*(.+)", abs_val, re.I)
            if m:
                body = m.group(1).strip()
                return {"value": [k.strip() for k in re.split(r"[,，;；、]\s*", body) if k.strip()],
                        "raw": body}
        return {"value": [], "raw": None}

    def _references(self, text, sections):
        sec = next((s for s in sections if s["kind"] == "refs"), None)
        if sec is None:
            sec = next((s for s in sections
                        if re.search(r"参考文献|^References?$|Bibliography", s["title"] or "", re.I)), None)
        items, title = [], None
        if sec is not None:
            title = sec["title"]
            raw = "\n".join(_strip_md(ln) for ln, _, _ in sec["lines"])
            items = self._split_numbered(raw)
            if not items:
                # 没有 [1]/1. 编号的（CLIP、APA 这类作者-年份格式），
                # 既然已经定位到 references 小节了，小节里每个 "- " 项就是一条
                items = self._split_bullet(raw)
        if not items:
            # 没有参考文献小节时，在全文里找 [1] [2] ... 条目。
            # 注意要支持列表符号，否则 pymupdf4llm 的 "- [1] xxx" 一条都匹配不上。
            cand = []
            for l in _lines(text):
                s = _strip_md(l)
                s = _LIST_MARK.sub("", s)
                if _REF_ENTRY.match(s):
                    cand.append(s)
            items = self._split_numbered("\n".join(cand))
        citations = sorted({c for c in _REF_CITATION.findall(text)},
                           key=lambda s: _numbers_in(s) or [0])
        return {"title": title, "items": items, "citations": citations}

    @staticmethod
    def _split_numbered(raw):
        """把一段文本按 [1]/[2]/[n]/1. 条目切开（兼容行首的 markdown 列表符号）。"""
        items, buf = [], []
        for ln in _lines(raw):
            t = ln.strip()
            if not t:
                continue
            body = _LIST_MARK.sub("", t)          # "- [1] xxx" -> "[1] xxx"
            if not body:
                continue
            if _SUBSECTION_NUM.match(body):       # "2.1 基于规则的方法" 是章节，不是文献
                if buf:
                    items.append("\n".join(buf).strip())
                    buf = []
                continue
            if _REF_ENTRY.match(body) and buf:
                items.append("\n".join(buf).strip())
                buf = [body]
            elif _REF_ENTRY.match(body) or buf:
                buf.append(body)
        if buf:
            items.append("\n".join(buf).strip())
        return [it for it in items if it]

    @staticmethod
    def _split_bullet(raw):
        """按行首的 markdown 列表符号切参考文献条目。

        _split_numbered 只认 "[1]" / "1." 这种编号。
        CLIP、APA 这类是「作者, 年份. 标题」开头，一条编号都没有，
        但在 references 小节里每条都是一个 "- " 列表项，按列表符号切就够了。
        """
        items, buf = [], []
        for ln in _lines(raw):
            t = ln.strip()
            if not t:
                continue
            if _LIST_MARK.match(t):
                if buf:
                    items.append("\n".join(buf).strip())
                buf = [_LIST_MARK.sub("", t)]
            elif buf:
                # 续行：上一条被 PDF 折断了，接到当前条目后面
                buf.append(t)
        if buf:
            items.append("\n".join(buf).strip())
        return [it for it in items if it]

    def _appendices(self, sections):
        out = []
        for sec in sections:
            if sec["kind"] != "appendix":
                continue
            out.append({
                "title": sec["title"],
                "page": sec["page"],
                "content": "\n".join(_strip_md(ln) for ln, _, _ in sec["lines"]).strip(),
            })
        return out

    # ---- 表格 ----
    def _tables(self, text, sections):
        tables = []
        lines = _lines(text)
        used = set()
        # 1) markdown 管道表
        i = 0
        while i < len(lines):
            if "|" in lines[i] and i + 1 < len(lines) and _MD_TABLE_SEP.match(lines[i + 1]):
                start = i
                rows = [_split_row(lines[i])]
                j = i + 2
                while j < len(lines) and "|" in lines[j]:
                    rows.append(_split_row(lines[j]))
                    j += 1
                caption, cap_page = self._caption_near(lines, start, used, "table")
                tables.append({
                    "type": "markdown",
                    "caption": caption,
                    "number": None,
                    "page": cap_page,
                    "n_rows": len(rows),
                    "n_cols": max(len(r) for r in rows) if rows else 0,
                    "header": rows[0] if rows else [],
                    "rows": rows,
                })
                used.update(range(start, j))
                i = j
                continue
            i += 1
        # 2) 只有题注没有表格线（扫描件/图片表）
        cap_lines, _ = self._scan_captions(lines, "table")
        covered = {t["caption"] for t in tables if t["caption"]}
        for cap in cap_lines:
            if cap["full"] in covered:
                continue
            tables.append({
                "type": "caption_only",
                "caption": cap["full"],
                "number": cap["number"],
                "page": cap["page"],
                "n_rows": None, "n_cols": None, "header": [], "rows": [],
                "context": self._context_around(lines, cap["line_no"], self.caption_window, used),
            })
        return tables

    # ---- 图 ----
    def _figures(self, text, sections):
        figures = []
        lines = _lines(text)
        used = set()
        for i, ln in enumerate(lines):
            for m in _MD_IMAGE.finditer(ln):
                alt, src = m.group(1).strip(), m.group(2).strip()
                if src.startswith("data:") or len(src) > 300:
                    src = src[:60] + "…(base64)"
                caption, cap_page = self._caption_near(lines, i, used, "figure")
                figures.append({
                    "type": "image",
                    "caption": caption or (alt or None),
                    "number": None,
                    "page": cap_page,
                    "src": src,
                    "alt": alt or None,
                })
                used.add(i)
        # 图片文字块（pymupdf4llm 的 OCR 图片层，扫描件的主要文字来源）
        for m in _PICTURE_TEXT_LOOSE.finditer(text):
            body = (m.group(1) or "").strip()
            if body:
                figures.append({"type": "picture_text", "caption": None, "number": None,
                                "page": None, "src": None, "alt": None, "text": body})
        # 只有图题
        covered = {f["caption"] for f in figures if f.get("caption")}
        for cap in self._scan_captions(lines, "figure")[0]:
            if cap["full"] in covered:
                continue
            figures.append({
                "type": "caption_only",
                "caption": cap["full"],
                "number": cap["number"],
                "page": cap["page"],
                "src": None, "alt": None,
                "context": self._context_around(lines, cap["line_no"], self.caption_window, used),
            })
        return figures

    def _scan_captions(self, lines, kind):
        """扫描图/表题注行。返回 (captions, 用到的行号集合)。"""
        pat = _CAPTION_TABLE if kind == "table" else _CAPTION_FIGURE
        caps, used = [], set()
        for i, ln in enumerate(lines):
            t = _strip_md(ln)
            m = pat.match(t)
            if not m:
                continue
            number, rest = m.group(1), (m.group(2) or "").strip()
            if not rest and len(t) > 60:
                continue
            caps.append({"number": number, "title": rest, "full": t, "line_no": i, "page": None})
            used.add(i)
        return caps, used

    def _caption_near(self, lines, idx, used, kind):
        """在给定行附近找图/表题注。"""
        caps, _ = self._scan_captions(lines, kind)
        best = None
        for cap in caps:
            d = cap["line_no"] - idx
            if -self.caption_window <= d <= self.caption_window:
                if best is None or abs(d) < abs(best[1]):
                    best = (cap, d)
        if best:
            used.add(best[0]["line_no"])
            return best[0]["full"], best[0]["number"]
        return None, None

    @staticmethod
    def _context_around(lines, idx, window, used):
        lo, hi = max(0, idx - window), min(len(lines), idx + window + 1)
        return "\n".join(_strip_md(lines[k]) for k in range(lo, hi) if k not in used).strip()

    # ---- 公式 ----
    def _formulas(self, text):
        formulas = []
        clean = _PAGE_TAG.sub("", text)
        # 1) $$...$$ / \[...\]
        for m in _FORMULA_BLOCK.finditer(clean):
            body = (m.group(1) or m.group(2) or "").strip()
            if not body:
                continue
            formulas.append({"type": "display", "value": _cap_math(body),
                             "number": self._formula_number(clean, m.end()),
                             "origin": "block"})
        # 2) 行内 $...$（排除纯变量）
        for i, ln in enumerate(_lines(clean)):
            for m in _FORMULA_INLINE.finditer(ln):
                body = m.group(1).strip()
                if not self._looks_like_formula(body):
                    continue
                formulas.append({"type": "inline", "value": _cap_math(body),
                                 "number": self._formula_number(clean, 0, line_no=i),
                                 "origin": "inline", "line": _strip_md(ln)[:200]})
        # 3) 编号写在公式之前的写法：'如下式(1)所示'（只取紧邻公式的短行或块级公式前一行）
        lines_all = _lines(clean)
        for i, ln in enumerate(lines_all):
            t = _strip_md(ln)
            if len(t) > 200:
                continue
            for m in re.finditer(r"(?:式|Eq\.?|Equation)\s*\(?\s*(\d{1,3})\s*\)?", t):
                nxt = _strip_md(lines_all[i + 1]) if i + 1 < len(lines_all) else ""
                if self._looks_like_formula(nxt):
                    formulas.append({"type": "reference", "value": _cap_math(nxt),
                                     "number": m.group(1), "origin": "pre_numbered",
                                     "line": t[:200]})
        # 4) 右侧编号行兜底（如 "E = mc^2    (3)"）
        for i, ln in enumerate(_lines(clean)):
            t = _strip_md(ln)
            m = re.match(r"^(.*?)\s*(?:\(\s*(\d{1,3})\s*\)|式\s*\(?\s*(\d{1,3})\s*\)?)\s*$", t)
            if not m:
                continue
            body = (m.group(1) or "").strip()
            num = m.group(2) or m.group(3)
            if len(body) < 3 or len(body) > 200 or not self._looks_like_formula(body):
                continue
            if any(body == f["value"] for f in formulas):
                continue
            formulas.append({"type": "numbered_line", "value": _cap_math(body),
                             "number": num, "origin": "numbered_line"})
        # 去重（保留首次出现）
        seen, out = set(), []
        for f in formulas:
            key = (f["type"], f["value"])
            if key in seen:
                continue
            seen.add(key)
            out.append(f)
        return out

    @staticmethod
    def _looks_like_formula(body):
        if not body or len(body) < 2 or len(body) > 300:
            return False
        if re.match(r"^[A-Za-z]$", body):     # 单个变量不算公式
            return False
        if re.search(r"[A-Za-z]", body) and not re.search(r"[=+^_\\{}<>≤≥∑∫√]|\d", body):
            return False
        return bool(re.search(r"[=+^_\\{}<>≤≥∑∫√]|\d\s*[A-Za-z]|[A-Za-z]\s*\d", body))

    @staticmethod
    def _formula_number(text, end, line_no=None):
        if line_no is not None:
            lines = _lines(text)
            tail = lines[line_no] if line_no < len(lines) else ""
            m = re.search(r"\(\s*(\d{1,3})\s*\)\s*$", tail)
            return m.group(1) if m else None
        window = text[end:end + 80]
        m = re.match(r"\s*(?:\(\s*(\d{1,3})\s*\)|式\s*\(?\s*(\d{1,3})\s*\)?)", window)
        return (m.group(1) or m.group(2)) if m else None

    # ---- 标题树 ----
    @staticmethod
    def _headings(sections):
        tree = []
        for sec in sections:
            tree.append({
                "title": sec["title"],
                "level": sec["level"],
                "number": ".".join(str(x) for x in sec["num"]) or None,
                "parent": sec["parent"],
                "kind": sec["kind"],
                "page": sec["page"],
            })
        return {
            "level1": [h for h in tree if h["level"] == 1],
            "level2": [h for h in tree if h["level"] == 2],
            "level3": [h for h in tree if h["level"] >= 3],
            "tree": tree,
        }

    # ---- 缺失提示 ----
    @staticmethod
    def _warnings(doc, sections, title, authors, abstract, tables, figures,
                  formulas, refs, appendices):
        w = []
        if not sections:
            w.append("未识别到任何标题（可能是扫描件/证书/图片型 PDF，正文缺少 markdown 标题标记）")
        if not (title or {}).get("value"):
            w.append("未识别到标题")
        if not authors.get("value"):
            w.append("未识别到作者")
        if not abstract.get("value"):
            w.append("未识别到摘要")
        if not tables:
            w.append("未识别到表格")
        if not figures:
            w.append("未识别到图")
        if not formulas:
            w.append("未识别到公式")
        if not refs:
            w.append("未识别到参考文献")
        if not appendices:
            w.append("未识别到附录")
        pages_span = (doc.get("page_start") or 1, doc.get("page_end") or 1)
        if pages_span[1] > pages_span[0]:
            w.append("文档共 %d 页（%s-%s），长文档的字段可能分布在多页"
                     % (doc.get("page_count") or 0, pages_span[0], pages_span[1]))
        return w


# ============================ 五、对外接口 ============================

class PaperExtractor:
    """
    对外主类：数据来源与 vctor_story_search 完全一致，但不依赖大模型。

        ex = PaperExtractor()                    # 自己建向量库连接
        ex = PaperExtractor(vector=tree.vector)  # 复用已有连接
        ex.get_all()                             # 全部文件：{文件名: {结构化字段}}
        ex.get_designate("四级.pdf")             # 指定文件

    取数优先级：
        1) 入库时写入的结构化结果（config.file_path_structured）：内容完整、页码准、快；
        2) 没有记录（老数据 / 入库时提取失败）时，回退到"从 Chroma 切块反推"的旧路径。
    """

    def __init__(self, vector=None, config=None):
        self.vector = vector or self._default_vector()
        self.parser = DocumentParser(config)
        self._assembler = None
        self._cache = {}
        # 入库时落盘的结构化结果缓存
        self._structured = {}
        self._structured_mtime = None

    @staticmethod
    def _default_vector():
        from vctor import vctor_story_search
        return vctor_story_search()

    # ---- 入库时落盘的结构化结果 ----
    def _load_structured(self):
        """读 config.file_path_structured；文件被入库流程改写过就重读（按 mtime 判断）。"""
        path = getattr(config, "file_path_structured", None)
        if not path or not os.path.exists(path):
            self._structured, self._structured_mtime = {}, None
            return self._structured
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            mtime = None
        if mtime == self._structured_mtime:
            return self._structured
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            docs = payload.get("documents", {}) if isinstance(payload, dict) else {}
            self._structured = docs if isinstance(docs, dict) else {}
        except (OSError, ValueError) as e:
            print(f"[paper_extractor] 结构化结果读取失败，回退切块解析: {e}")
            self._structured = {}
        self._structured_mtime = mtime
        return self._structured

    def _structured_get(self, filename):
        return self._load_structured().get(filename)

    # ---- 取数 ----
    def raw_all(self):
        return self.vector.get_all()

    def raw_designate(self, filename):
        return self.vector.get_designate(filename)

    def list_sources(self):
        """优先用结构化记录的文件名；再并入只在向量库里的文件。"""
        names = set(self._load_structured().keys())
        if self._assembler is None:
            self._assembler = ChunkAssembler(self.raw_all())
        names.update(self._assembler.sources())
        return sorted(names)

    def _parse_docs(self, docs):
        asm = ChunkAssembler(docs)
        out = {}
        for source in asm.sources():
            out[source] = self._parse_one(asm, source)
        return out

    def _parse_one(self, asm, source):
        if source in self._cache:
            return self._cache[source]
        doc = asm.build(source)
        if doc is None:
            return None
        parsed = self.parser.parse(doc)
        parsed["pages"] = doc["pages"]
        parsed["chunk_ids"] = [c["id"] for c in doc["chunks"]]
        parsed["chunk_metadatas"] = doc["metadatas"]
        self._cache[source] = parsed
        return parsed

    # ---- 主要方法 ----
    def get_all(self):
        """全部文件的结构化信息：优先入库时的完整内容结果，缺的用切块反推补齐。"""
        result = dict(self._load_structured())
        cached = {k for k in result if k in self._cache}
        asm = ChunkAssembler(self.raw_all())
        for source in asm.sources():
            if source in result:
                if source in self._cache:
                    result[source] = self._cache[source]
                continue
            item = self._parse_one(asm, source)
            if item is not None:
                result[source] = item
        return result

    def get_designate(self, filename):
        """指定文件的结构化信息（无该文件返回 None）。"""
        hit = self._structured_get(filename)
        if hit is not None:
            return hit
        return self._parse_docs(self.raw_designate(filename)).get(filename)

    def get(self, filename=None, include_sections=True):
        """统一入口：filename 为空＝全库。"""
        if filename is None:
            result = self._parse_docs(self.raw_all())
            if not include_sections:
                for v in result.values():
                    v.pop("sections", None)
            return result
        result = self.get_designate(filename)
        if result is not None and not include_sections:
            result.pop("sections", None)
        return result

    def get_table(self, filename=None):
        """只取表格：指定文件返回 list，否则返回 {文件名: list}。"""
        data = self.get(filename, include_sections=False)
        if filename is not None:
            return data["tables"] if data else []
        return {k: v["tables"] for k, v in data.items()}

    def get_heading(self, filename=None):
        """只取标题层级。"""
        data = self.get(filename, include_sections=False)
        if filename is not None:
            return data["headings"] if data else {}
        return {k: v["headings"] for k, v in data.items()}

    def get_formula(self, filename=None):
        """只取公式。"""
        data = self.get(filename, include_sections=False)
        if filename is not None:
            return data["formulas"] if data else []
        return {k: v["formulas"] for k, v in data.items()}

    def summary(self, filename=None):
        """精简总览：标题/作者/摘要长度/各级标题数/表图公式参考文献附录计数。"""
        data = self.get(filename, include_sections=False)
        items = data if filename is None else ({filename: data} if data else {})

        def one(d):
            if not d:
                return None
            h = d["headings"]
            return {
                "title": (d["title"] or {}).get("value"),
                "authors": (d["authors"] or {}).get("value"),
                "abstract_chars": len((d["abstract"] or {}).get("value") or ""),
                "level1": len(h["level1"]), "level2": len(h["level2"]), "level3": len(h["level3"]),
                "tables": len(d["tables"]), "figures": len(d["figures"]),
                "formulas": len(d["formulas"]), "references": d["references"]["count"],
                "appendices": len(d["appendices"]),
                "pages": "%s-%s" % (d.get("page_start"), d.get("page_end")),
            }
        return {k: one(v) for k, v in items.items()}

    def extract(self, filename=None, to_json=False, indent=2, include_sections=True,
                save_path=None):
        """filename 为空＝提取全部；to_json/save_path 用于导出 JSON。"""
        data = self.get(filename, include_sections=include_sections)
        if not to_json and not save_path:
            return data
        text = json.dumps(data, ensure_ascii=False, indent=indent, default=str)
        if save_path:
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(text)
            return save_path
        return text


# ---- 不依赖向量库的便捷入口（可用手写 markdown 单测） ----
def extract_from_docs(docs, config=None, include_sections=True):
    """直接吃 vctor 风格的 {"ids","documents","metadatas"}，返回 {文件名: 结构化结果}。"""
    parser = DocumentParser(config)
    asm = ChunkAssembler(docs)
    out = {}
    for source in asm.sources():
        doc = asm.build(source)
        parsed = parser.parse(doc)
        parsed["pages"] = doc["pages"]
        parsed["chunk_ids"] = [c["id"] for c in doc["chunks"]]
        parsed["chunk_metadatas"] = doc["metadatas"]
        if not include_sections:
            parsed.pop("sections", None)
        out[source] = parsed
    return out


# ---- 入库阶段（完整分页内容，无需从切块反推） ----

def extract_from_pages(data, filename, parser=None):
    """入库时直接用 pymupdf4llm 的分页结果提取结构化信息。

    data 就是 knolege._parse_pdf_base64() 返回的
        [{"text": ..., "metadata": {"page_number": N}}, ...]
    页序天然正确、内容完整，因此不需要 ChunkAssembler 那套"按 (页, 页内序号)
    排序 + 插入分页标记"的还原逻辑，也不会遇到跨块截断。

    返回与 extract_from_docs 完全相同的字段结构。
    """
    data = list(data or [])
    page_texts, metas = [], []
    for i, page in enumerate(data):
        if isinstance(page, dict):
            text = page.get("text") or ""
            meta = page.get("metadata") or {}
        else:
            text, meta = str(page or ""), {}
        if not isinstance(meta, dict):
            meta = {}
        page_no = meta.get("page_number")
        page_texts.append(text)
        metas.append({
            "source": filename,
            "page_number": page_no if page_no is not None else i + 1,
            "create_time": meta.get("create_time", ""),
        })

    pages = {}
    for text, meta in zip(page_texts, metas):
        pages.setdefault(meta["page_number"], []).append(text)

    numbered = [m["page_number"] for m in metas] or [1]
    # 插入分页标记：DocumentParser 靠它把每一行/每个元素归属到准确页码。
    # 这里和"从切块反推"不同——页序本来就对，标记只用于标注归属，规则见 _PAGE_TAG。
    marked_text = "\n\n".join(
        "<!-- page:%s -->\n%s" % (meta["page_number"], text)
        for meta, text in zip(metas, page_texts)
    )
    doc = {
        "source": filename,
        "text": marked_text,
        "page_start": min(numbered),
        "page_end": max(numbered),
        "page_count": len(set(numbered)),
        "pages": {p: "\n".join(t) for p, t in sorted(pages.items())},
        "chunks": [],
        "metadatas": metas,
    }
    parsed = (parser or DocumentParser()).parse(doc)
    parsed["pages"] = doc["pages"]
    return parsed


def load_structured(path=None):
    """读结构化结果文件，返回 {文件名: 结构化结果}。"""
    path = path or getattr(config, "file_path_structured", None)
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, ValueError) as e:
        print(f"[paper_extractor] 结构化结果读取失败: {e}")
        return {}
    docs = payload.get("documents", {}) if isinstance(payload, dict) else {}
    return docs if isinstance(docs, dict) else {}


def save_structured(structured, path=None):
    """写结构化结果文件（整体覆盖）。"""
    path = path or getattr(config, "file_path_structured", None)
    if not path:
        raise ValueError("未配置 config.file_path_structured")
    payload = {
        "version": STRUCT_VERSION,
        "create_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "documents": structured or {},
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    return path


def upsert_structured(filename, info, path=None):
    """新增/覆盖单个文件的结构化结果（入库时调用）。"""
    structured = load_structured(path)
    structured[filename] = info
    save_structured(structured, path)
    return structured


def drop_structured(filename, path=None):
    """删除单个文件的结构化结果（删文档时调用），返回被删除的内容。"""
    structured = load_structured(path)
    removed = structured.pop(filename, None)
    if removed is not None:
        save_structured(structured, path)
    return removed


if __name__ == "__main__":
    import sys
    target = sys.argv[1] if len(sys.argv) > 1 else None
    ex = PaperExtractor()
    print(json.dumps(ex.summary(target), ensure_ascii=False, indent=2))
    if target:
        print(json.dumps(ex.get_designate(target), ensure_ascii=False, indent=2, default=str))
