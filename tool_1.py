# -*- coding: utf-8 -*-
"""论文翻译：按 sections 一节一节翻，译文写 translate.json

翻译单位用 chroma_structured.json 里的 sections，不按页也不按切块：
页边界和切块边界都会把句子从中间切断，sections 是入库时按标题层级切的，
每节自带 title 和 page，正好能对上「第几页」。

流程：
    sections[i].content → 删掉图里的 OCR 乱码 → 表格抽成 [[BLOCK_0]]
    → 调模型 → 表格拼回去 → translate.json

对外两个东西：
    translate(filename, section=None)   干活的核心函数
    translate_paper                      @tool 包了一层，塞给 agent 当工具
"""
import hashlib
import json
import os
import re
import time

from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

import config

# 和 vctor.py / rag.py 一样，.env 就在本文件旁边
load_dotenv()


# ============================ 两个文件长什么样 ============================
#
# 输入 config.file_path_structured（入库时 paper_extractor 写的）：
#   {"documents": {
#       "1706.03762v1.pdf": {
#           "title": "...", "authors": [...], "tables": [...],
#           "sections": [
#               {"title": "Abstract", "level": 1, "page": 1,
#                "content": "The dominant sequence transduction models ..."},
#               {"title": "References", "level": 1, "page": 10, "content": "- [1] ..."},
#               ...]}}}
#
# 输出 RESULT_PATH（本文件写的，结构和上面并列）：
#   {"version": 1,
#    "documents": {
#        "1706.03762v1.pdf": {
#            "source": "1706.03762v1.pdf", "model": "qwen3.8-max",
#            "sections": {
#                "Abstract": {"page": 1, "zh": "占主导地位的序列转换模型..."}}}}}
#
# 中间 CACHE_PATH：{"1706.03762v1.pdf::Abstract::v1::a3400480": "占主导地位的..."}
# =========================================================================


MODEL = "qwen3.8-max"
PROMPT_VERSION = "v1"          # 改 prompt 就加一，缓存自动失效
CACHE_PATH = r"./translate_cache.json"    # 缓存：翻译过的段落不必再花钱
RESULT_PATH = r"./translate.json"         # 成品：给前端和 agent 用

# 需求文档明确要求：参考文献不译，只保留标题
SKIP = ("references", "bibliography")

# pymupdf 会把图里的字 OCR 出来塞进正文，形如
#   <!-- Start of picture text -->  Linear<br>MatMul<br>Concat  <!-- End of picture text -->
# 不删的话模型会去硬翻这堆框图乱码，既浪费 token 又破坏译文
PIC_RE = re.compile(r"<!--\s*Start of picture text\s*-->.*?<!--\s*End of picture text\s*-->", re.S)

# 术语表：翻译质量最大的杀手是同一个词前后翻法不同，
# 所以固定一张表塞进 prompt，比事后人工统一省事得多
GLOSSARY = r"""attention → 注意力
self-attention → 自注意力
multi-head attention → 多头注意力
Transformer / BLEU / dropout → 不译，保留英文
encoder / decoder → 编码器 / 解码器
positional encoding → 位置编码
residual connection → 残差连接
layer normalization → 层归一化
feed-forward network → 前馈网络
sequence transduction → 序列转换"""

# {glossary} 在 __init__ 里被替换成上面的表。
# 规则 4/5/6 都是「必须原样照抄」的东西，少一条译文就会坏：
#   4 数学符号 → 不照抄的话 _dk_ 会被翻成「dk 的维度」
#   5 引用编号 → 不照抄的话 [34] 会被翻成「第三十四篇」或者直接删掉
#   6 占位符   → 不照抄的话表格就丢了（模型很爱把 [[BLOCK_0]] 改掉）
SYSTEM_PROMPT = r"""你是学术论文翻译专家，把英文学术论文准确译成中文。

【规则】
1. 只输出译文，不要「以下是译文」这类开场白，不要解释、总结、译注。
2. 忠实原文：不增不减，不合并不拆分句子，不改顺序。
3. 保留段落结构，段与段之间空一行。
4. 数学符号原样照抄：下划线变量 _dk_、上标 <sup>..</sup>、运算符 · × √ ϵ。
5. 引用编号原样照抄：[34]、[11]。
6. 遇到 [[BLOCK_0]] 这样的占位符，代表一张表格或一段代码：
   原样保留、单独一行、不翻译、不改括号、不移动位置。
7. 术语按下表，全文一致：
{glossary}
8. 用书面学术语体，不要口语化。

只输出译文。"""


class Translator(object):
    def __init__(self):
        self.llm = ChatOpenAI(
            model=MODEL,
            base_url=os.getenv("qwen_url"),
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            temperature=0.2,        # 翻译要的是稳，不是活，所以压低
        )
        self.system = SYSTEM_PROMPT.replace("{glossary}", GLOSSARY)

        # 缓存整个读到内存，之后只在内存里查，写的时候才落盘
        if os.path.exists(CACHE_PATH):
            with open(CACHE_PATH, encoding="utf-8") as f:
                self.cache = json.load(f)
        else:
            self.cache = {}

    def prepare(self, text):
        """删图里的 OCR 乱码，再把表格抽成 [[BLOCK_0]] 占位符

        表格为什么不直接发给模型：模型改表格是常态（少一列、吃掉下划线、
        把 <sup> 展开），Table 3 有 34 行还白占 token。
        抽出来翻完再按位置拼回去，100% 保真。

        抽之前：                        抽之后（发给模型的是这个）：
            |Model|BL|EU|                  [[BLOCK_0]]
            |---|---|
            |GNMT + RL [34]|24.6|39.92|

        返回 (掩码后的正文, 抽出来的表格列表)，表格顺序就是 [[BLOCK_0]]、
        [[BLOCK_1]] 的顺序，拼回去靠这个下标对上。
        """
        lines = PIC_RE.sub("", text).split("\n")
        out = []        # 掩码后的行
        blocks = []     # 被抽出来的表格原文
        i = 0
        while i < len(lines):
            # lstrip 是必须的：表格行前面偶尔有空格，不 lstrip 会漏判
            if lines[i].lstrip().startswith("|"):
                # 连续的 | 开头行算一整张表，一直往后吃到不是表格为止
                j = i
                while j < len(lines) and lines[j].lstrip().startswith("|"):
                    j += 1
                blocks.append("\n".join(lines[i:j]))
                # 整张表换成一个占位符，单独占一行，位置留在原地
                out.append("[[BLOCK_%d]]" % (len(blocks) - 1))
                i = j
            else:
                # 非表格行原样留着，rstrip 去掉行尾空格省点 token
                out.append(lines[i].rstrip())
                i += 1
        return "\n".join(out).strip(), blocks

    def ask(self, title, page, text):
        """调一次模型，返回译文

        重试 3 次是必要的：DashScope 免费额度会 429，网络也会抖，
        一次失败就整节丢掉太亏。退避 1s → 2s → 4s，别把限流撞得更狠。

        user 里带【节标题】和【页码】不是给模型看的，是给它「定位」用的：
        告诉它这是哪一节，译文才不会自己编上下文（比如把 Abstract 译成引言）。
        """
        for n in range(3):
            try:
                msg = self.llm.invoke([
                    ("system", self.system),
                    ("user", "【节标题】%s\n【页码】%s\n\n%s" % (title, page, text)),
                ])
                return msg.content.strip()
            except Exception as e:
                print("  第 %d 次失败：%s" % (n + 1, e))
                time.sleep(2 ** n)
        # 三次都失败就抛，不能返回空串让下游把空译文当成翻译成功
        raise RuntimeError("%s 翻译失败" % title)

    def translate(self, filename, section=None):
        """翻一篇或一节，返回 {节标题: {"page": 页码, "zh": 译文}}

        filename 传文件名，要和库里一致，如 "1706.03762v1.pdf"
        section  传节标题，如 "Abstract"；None 就是整篇
        """
        # 每次现读，不用缓存：重新入库后 sections 会变，读旧的会翻到过期的正文
        with open(config.file_path_structured, encoding="utf-8") as f:
            docs = json.load(f)["documents"]
        sections = docs[filename]["sections"]

        # 只翻指定的一节。全等匹配，不做模糊：模糊匹配会把
        # "Attention" 同时命中 "Attention" 和 "Scaled Dot-Product Attention"
        if section:
            hit = []
            for sec in sections:
                if sec["title"].lower() == section.lower():
                    hit.append(sec)
            sections = hit

        result = {}
        for sec in sections:
            title = sec["title"].strip()

            # 参考文献不译（需求文档 1.3 的要求）
            if title.lower() in SKIP:
                continue

            # content 是空的就是纯标题节，比如这篇的 "Results"，
            # 正文其实在它的子节 20/22 里，这一节本身没东西可翻
            content = sec["content"]
            if not content:
                continue

            masked, blocks = self.prepare(content)

            # 清洗完可能什么都不剩（整节就是一张图），那就别浪费一次调用
            if not masked:
                continue

            # 缓存键 = 谁 + 哪节 + prompt 版本 + 内容指纹
            # 带可读前缀是为了打开 translate_cache.json 能看出是哪篇哪节，
            # 纯 md5 的话出问题根本没法查。
            # 指纹里必须含表格原文：表格变了而正文没变时，如果只 hash 正文
            # 就会一直命中旧译文，表格永远停在旧版本
            raw = MODEL + "\n" + PROMPT_VERSION + "\n" + masked
            for block in blocks:
                raw = raw + "\n" + block
            digest = hashlib.md5(raw.encode("utf-8")).hexdigest()[:8]
            key = filename + "::" + title + "::" + PROMPT_VERSION + "::" + digest

            if key in self.cache:
                zh = self.cache[key]
                print("缓存：%s" % title)
            else:
                zh = self.ask(title, sec["page"], masked)
                # 每翻完一节立刻落盘：翻到第 20 节崩了，前 19 节的钱不白花
                self.cache[key] = zh
                with open(CACHE_PATH, "w", encoding="utf-8") as f:
                    json.dump(self.cache, f, ensure_ascii=False, indent=2)
                print("翻译：%s" % title)

            # 表格拼回去。正常情况占位符还在，直接 replace；
            # 模型偶尔会把 [[BLOCK_0]] 吃掉或改写法，这时补在末尾，
            # 宁可位置难看也不能把整张表弄丢
            for i in range(len(blocks)):
                mark = "[[BLOCK_%d]]" % i
                if mark in zh:
                    zh = zh.replace(mark, blocks[i])
                else:
                    zh = zh + "\n\n" + blocks[i]

            result[title] = {"page": sec["page"], "zh": zh}

        self.save(filename, result)
        return result

    def save(self, filename, result):
        """按节合并写入 translate.json

        为什么要合并：先只翻 Abstract 验证，再翻全篇是很正常的两步走。
        如果 save 直接覆盖，第二次只翻别的节时会把 Abstract 冲掉。
        """
        if os.path.exists(RESULT_PATH):
            with open(RESULT_PATH, encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = {"version": 1, "documents": {}}

        # 先把这篇已有的译文取出来，再拿本次结果覆盖同名的节
        old = {}
        if filename in data["documents"]:
            old = data["documents"][filename]["sections"]
        for title in result:
            old[title] = result[title]

        data["documents"][filename] = {"source": filename, "model": MODEL, "sections": old}
        with open(RESULT_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, filename):
        """读已翻好的结果，给前端用，不触发翻译

        前端渲染中英对照时：英文在 chroma_structured.json 的 sections，
        中文在这里；两边都用 title 做键，直接按标题对上就行。
        """
        if not os.path.exists(RESULT_PATH):
            return {}
        with open(RESULT_PATH, encoding="utf-8") as f:
            data = json.load(f)
        if filename not in data["documents"]:
            return {}
        return data["documents"][filename]


# 懒加载单例：import 本模块时不连模型，第一次真正翻译才建
translator = None


def translate(filename, section=None):
    """翻一篇或一节。section 传节标题如 "Abstract"，不传就是全篇"""
    global translator
    if translator is None:
        translator = Translator()
    return translator.translate(filename, section)


@tool
def translate_paper(filename: str, section: str = ""):
    """把本地文献库中的英文论文翻译成中文。

    用户说「翻译一下这篇」「这段什么意思」时用这个工具。
    本工具只给译文，不回答内容问题，问内容请用 local_search。

    Args:
        filename: 论文文件名，要和库里一致，如 1706.03762v1.pdf
        section: 只翻某一节，如 Abstract、Conclusion；留空表示全篇
    """
    # 这个 if 不能省：LangChain 的 tool 会把缺省值原样传进来，
    # 传进来的空串要换成 None，否则 translate 里会去匹配空标题
    if section:
        data = translate(filename, section)
    else:
        data = translate(filename)

    # 模型只认字符串，所以把 {标题: {page, zh}} 拼成一段带页码的文本。
    # 页码一定要带上：用户问「这句在哪页」时模型只能从这里拿到答案
    parts = []
    for title in data:
        item = data[title]
        parts.append("【%s】第 %s 页\n%s" % (title, item["page"], item["zh"]))
    return "\n\n".join(parts)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("用法: python tool_1.py 文件名 [节标题]")
        sys.exit(1)
    if len(sys.argv) > 2:
        out = translate(sys.argv[1], sys.argv[2])
    else:
        out = translate(sys.argv[1])
    for title in out:
        print("\n" + "=" * 60)
        print("【%s】第 %s 页" % (title, out[title]["page"]))
        print(out[title]["zh"])
