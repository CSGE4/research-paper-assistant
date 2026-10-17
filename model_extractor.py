# -*- coding: utf-8 -*-
"""
model_extractor.py —— 用大模型补规则解析的缺口。

为什么需要它（而不是继续加正则）：
    规则解析是"预判 PDF 会怎么排版"，而排版是开放集合 —— 换个模板就要重写正则。
    标题/作者这两个字段尤其如此（加粗、井号、居中、图片层……写法无穷）。
    所以这里的分工是：
        规则负责有强锚点的字段（表格/公式/参考文献编号）—— 快、免费、确定；
        模型负责排版多变的字段（标题/作者/关键词等）—— 灵活，但必须防编造。

三条硬约束：
    1) 只送最小上下文：标题/作者只需第一页；参考文献只需含 References 的页。
       绝不整篇丢给模型 —— 省 token，也降低编造机会。
    2) 防幻觉：模型返回的每个值都必须能在原文中找到（归一化后做包含检查），
       找不到的一律丢弃，宁可留空也不写错。
    3) 不盲信：规则的"错值"（比如把 "1 Introduction" 当标题）也要能被模型纠正，
       但只有模型答案通过校验、且明显比规则结果更像该字段时才采纳。

开关：
    MODEL_EXTRACT_ENABLED=0     关闭模型补全（只用规则）
    MODEL_EXTRACT_MODELS=a,b    模型链，按顺序尝试，默认 qwen3.8-max,deepseek-chat
"""

import json
import os
import re

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from paper_extractor import (
    _PAGE_TAG, _lines, _has_html_comment, _norm_text,
)

load_dotenv()

# ---------------------------------------------------------------- 配置

# 模型兜底负责的字段（其余字段规则更可靠，不动）
MODEL_FIELDS = ("title", "authors", "keywords")

# 默认模型链：先 DashScope 的 qwen3.8-max，失败退到 DeepSeek
DEFAULT_CHAIN = ("qwen3.8-max", "deepseek-chat")

# 送进模型的正文字符上限（防止 prompt 过长 / 费用失控）
MAX_CONTEXT_CHARS = 6000


def _enabled():
    return os.getenv("MODEL_EXTRACT_ENABLED", "1").strip().lower() not in ("0", "false", "no", "off")


def _model_chain():
    raw = os.getenv("MODEL_EXTRACT_MODELS", "").strip()
    if raw:
        return tuple(m.strip() for m in raw.split(",") if m.strip())
    return DEFAULT_CHAIN


def _provider_conf(name):
    """按模型名判断用哪个 provider。"""
    if name.startswith("deepseek"):
        return os.getenv("deepseek_url"), os.getenv("deepseek_api_key")
    return os.getenv("qwen_url"), os.getenv("DASHSCOPE_API_KEY")


def build_model(name=None, temperature=0, timeout=90):
    """构造一个 LLM；name 为空时取模型链第一个。"""
    name = name or _model_chain()[0]
    base_url, api_key = _provider_conf(name)
    return ChatOpenAI(model=name, base_url=base_url, api_key=api_key,
                      temperature=temperature, timeout=timeout, max_retries=1)


# ---------------------------------------------------------------- 缺口识别

def weak_fields(info):
    """返回"规则不可靠"的字段名列表。

    满足任一条件即视为弱：
      - 规则没取到（空值）
      - 规则取到的是疑似错值（例如标题其实是一个编号章节名）
    """
    weak = []
    info = info or {}

    t = ((info.get("title") or {}).get("value") or "").strip()
    if not t or _looks_like_section(t):
        weak.append("title")

    au = ((info.get("authors") or {}).get("value") or [])
    if not au:
        weak.append("authors")

    kw = ((info.get("keywords") or {}).get("value") or [])
    if not kw:
        weak.append("keywords")

    return weak


def _looks_like_section(t):
    """标题看起来是"章节名"而不是文章名（1 Introduction / 2 相关工作 / Abstract）。"""
    if not t:
        return True
    if re.match(r"^\s*\d{1,2}(?:\.\d{1,2})*\.?\s+\S", t):
        return True
    if re.match(r"^\s*(摘\s*要|Abstract|ABSTRACT|引\s*言|绪\s*论|前\s*言|"
                r"结\s*论|参考文献|References?|致\s*谢|附\s*录)\s*$", t, re.I):
        return True
    return False


# ---------------------------------------------------------------- 上下文

def _compress(text, limit=MAX_CONTEXT_CHARS):
    """把页面文本压缩到 limit 以内：按行去掉空行与版式注释。"""
    out, total = [], 0
    for ln in _lines(text):
        s = _PAGE_TAG.sub("", ln).strip()
        if not s or _has_html_comment(s):
            continue
        if total + len(s) + 1 > limit:
            break
        out.append(s)
        total += len(s) + 1
    return "\n".join(out)


def build_context(info, fields, per_field_limit=2600):
    """只取与所需字段相关的原文片段（第一页 + 参考文献所在页）。"""
    chunks = []
    pages = info.get("pages") or {}
    keys = sorted(pages.keys(), key=lambda x: (x is None, x))
    first = keys[0] if keys else None
    if first is not None:
        txt = _compress(pages.get(first) or "", per_field_limit)
        if txt:
            chunks.append("【第 %s 页】\n%s" % (first, txt))

    # 参考文献在最后几页：附上含 References 标题的那一页
    if "references" in fields:
        ref_page = None
        for h in (info.get("headings") or {}).get("tree", []):
            if h.get("kind") == "refs":
                ref_page = h.get("page")
                break
        if ref_page is None and keys:
            ref_page = keys[-1]
        if ref_page is not None and ref_page != first:
            txt = _compress(pages.get(ref_page) or "", per_field_limit)
            if txt:
                chunks.append("【第 %s 页】\n%s" % (ref_page, txt))

    return "\n\n".join(chunks)[:MAX_CONTEXT_CHARS]


# ---------------------------------------------------------------- 模型调用

_PROMPT = """你是文档信息抽取器。从下面的原文里抽取指定字段。

严格要求：
1. 只输出一个 JSON 对象，不要输出任何解释、注释、markdown 代码块标记。
2. 只抽取原文中明确出现的内容；原文里没有的字段，值用 null 或空数组，禁止编造。
3. title 是文章的正式标题（不是"1 Introduction""Abstract"这类章节名）。
4. authors 是作者姓名列表，不要机构名、不要邮箱、不要"∗"之类的符号。
5. keywords 是原文明确标注的关键词列表，没有就给空数组。

需要抽取的字段：{fields}
输出格式示例：{example}

原文：
{context}
"""

_EXAMPLE = {
    "title": {"value": "文章标题或 null"},
    "authors": {"value": ["作者1", "作者2"]},
    "keywords": {"value": ["关键词1"]},
}


def _extract_json(text):
    """从模型输出里抠出 JSON 对象（容忍 ```json 包裹 / 前后废话）。"""
    if not text:
        return None
    s = text.strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", s, re.S)
    if m:
        s = m.group(1).strip()
    else:
        start, end = s.find("{"), s.rfind("}")
        if start >= 0 and end > start:
            s = s[start:end + 1]
    s = s.replace("“", '"').replace("”", '"')
    try:
        return json.loads(s)
    except ValueError:
        return None


def call_model(context, fields, model=None, chain=None):
    """调用模型链，返回 (解析后的 dict, 实际使用的模型名) 或 (None, 错误信息)。"""
    fields = [f for f in fields if f in MODEL_FIELDS]
    if not fields or not context:
        return None, "no_fields_or_context"

    prompt = _PROMPT.format(
        fields=", ".join(fields),
        example=json.dumps({k: _EXAMPLE[k] for k in fields if k in _EXAMPLE},
                           ensure_ascii=False),
        context=context,
    )
    chain = list(chain) if chain else ([model] if model else list(_model_chain()))
    errors = []
    for name in chain:
        try:
            llm = build_model(name)
            resp = llm.invoke(prompt)
            content = resp.content if hasattr(resp, "content") else str(resp)
            data = _extract_json(content)
            if isinstance(data, dict):
                return data, name
            errors.append("%s: 返回不是合法 JSON" % name)
        except Exception as e:
            errors.append("%s: %s" % (name, str(e)[:160]))
    return None, " | ".join(errors)


# ---------------------------------------------------------------- 防幻觉校验

def _norm(s):
    return re.sub(r"\s+", " ", (s or "")).strip()


def _source_blob(info):
    """用于校验的原文（所有页拼起来，供包含性检查）。"""
    pages = info.get("pages") or {}
    keys = sorted(pages.keys(), key=lambda x: (x is None, x))
    return _norm(" \n ".join(str(pages.get(k) or "") for k in keys))


def verify(value, blob, kind="text"):
    """校验模型给的值是否真的出现在原文里。"""
    if value is None:
        return False
    if kind == "list":
        if not isinstance(value, list):
            return False
        return all(v and _norm(str(v)) in blob for v in value)
    v = _norm(str(value))
    if not v:
        return False
    return v in blob


# ---------------------------------------------------------------- 合并

def merge(info, model_out, blob=None):
    """把模型结果合进规则结果。

    返回 (新的 info, 明细)。明细记录每个字段的来源与是否与规则冲突，
    便于在页面/日志里看见"这个值是谁给的"，而不是黑箱。
    """
    blob = blob if blob is not None else _source_blob(info)
    info = dict(info or {})
    details = {}

    for field in MODEL_FIELDS:
        rule_val = _rule_value(info, field)
        got = model_out.get(field) if isinstance(model_out, dict) else None
        model_val = got.get("value") if isinstance(got, dict) else got

        # 1) 模型没给 → 保留规则
        if model_val in (None, "", [], {}):
            details[field] = {"source": "rule", "changed": False, "rule": rule_val}
            continue

        # 2) 防幻觉：模型值必须能在原文核对到
        kind = "list" if field in ("authors", "keywords") else "text"
        if not verify(model_val, blob, kind):
            details[field] = {"source": "rejected_hallucination", "changed": False,
                              "rule": rule_val, "model": model_val}
            continue

        # 3) 模型和规则一致 → 记为一致
        if _same(rule_val, model_val):
            details[field] = {"source": "both", "changed": False, "rule": rule_val}
            continue

        # 4) 不一致：标题这类字段，只要模型答案通过校验且更像标题就采用
        if _prefer_model(field, rule_val, model_val):
            _set_value(info, field, model_val)
            details[field] = {"source": "model", "changed": True,
                              "rule": rule_val, "model": model_val}
        else:
            details[field] = {"source": "rule", "changed": False,
                              "rule": rule_val, "model": model_val}

    info["extract_source"] = _summarize(details)
    info["field_details"] = details
    return info, details


def _rule_value(info, field):
    v = (info.get(field) or {})
    return v.get("value") if isinstance(v, dict) else v


def _set_value(info, field, value):
    cur = info.get(field)
    if isinstance(cur, dict):
        cur = dict(cur)
        cur["value"] = value
        cur["origin"] = "model"
        info[field] = cur
    else:
        info[field] = {"value": value, "origin": "model"}


def _same(a, b):
    if isinstance(a, list) and isinstance(b, list):
        return [_norm(str(x)) for x in a] == [_norm(str(x)) for x in b]
    return _norm(str(a)) == _norm(str(b))


def _prefer_model(field, rule_val, model_val):
    """是否用模型值覆盖规则值。"""
    if field == "title":
        if not rule_val:
            return True
        if _looks_like_section(rule_val):
            return True                     # 规则值明显是章节名
        # 规则值不是章节名时保守：只有模型答案明显更长才覆盖
        return len(_norm(str(model_val))) > len(_norm(str(rule_val))) * 1.5
    if field in ("authors", "keywords"):
        # 规则空 → 用模型；规则非空则保留规则（作者通常规则更准）
        return not rule_val
    return False


def _summarize(details):
    return {field: d.get("source") for field, d in details.items()}


# ---------------------------------------------------------------- 对外入口

def enrich(info, fields=None):
    """对一个已经用规则提取好的 info 做模型补全。

    这是唯一需要外部调用的函数：
        info = extract_from_pages(data, filename)   # 规则
        info = enrich(info)                          # 模型补缺
    失败时原样返回（绝不让模型问题影响主流程）。
    """
    if not info:
        return info
    try:
        if not _enabled():
            return info
        need = fields or weak_fields(info)
        need = [f for f in need if f in MODEL_FIELDS]
        if not need:
            info.setdefault("extract_source", {})
            return info
        context = build_context(info, need)
        if not context:
            return info
        model_out, used = call_model(context, need)
        if not isinstance(model_out, dict):
            info["extract_source"] = {"model_error": used}
            return info
        merged, details = merge(info, model_out)
        merged["model_used"] = used
        return merged
    except Exception as e:
        print(f"[model_extractor] 模型补全失败（保留规则结果）: {e}")
        return info


if __name__ == "__main__":
    import sys
    import pymupdf
    import pymupdf4llm
    from paper_extractor import extract_from_pages

    pdf = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\34078\Desktop\1706.03762v1.pdf"
    name = os.path.basename(pdf)
    doc = pymupdf.open(pdf)
    data = pymupdf4llm.to_markdown(doc, page_chunks=True)
    doc.close()
    before = extract_from_pages(data, name)
    print("规则结果: title=%r" % ((before.get("title") or {}).get("value"),))
    print("         authors=%s" % ((before.get("authors") or {}).get("value"),))
    print("弱字段:", weak_fields(before))
    after = enrich(before)
    print("补全后  : title=%r" % ((after.get("title") or {}).get("value"),))
    print("         authors=%s" % ((after.get("authors") or {}).get("value"),))
    print("字段来源:", after.get("extract_source"))
    print("使用的模型:", after.get("model_used"))
