from translation import Translator
# 懒加载单例：import 本模块时不连模型，第一次真正翻译才建
translator = None
def translate_paper(filename: str, section: str = ""):
    """把本地文献按章节翻译成中文，返回中文译文（含节标题和页码）。

    仅当用户明确要求「翻译 / 译成中文 / 中文版」时使用。
    用户问内容、问含义时不要用本工具，改用 local_search —— 本工具只给译文，不给结论。

    Args:
        filename: 论文文件名，必须与库中一致，如 1706.03762v1.pdf。
                  不确定时先调用一次，让报错信息列出可用文件，不要编造。
        section: 要翻译的节标题，如 Abstract、Introduction、Conclusion。
                 一次只传一节；用户要整篇时请按节分多次调用，不要留空。
    """
    global translator
    if translator is None:
        translator = Translator()

    # 不传节标题就等于要翻整篇，26 节一次返回会撑爆上下文，这里直接拦掉
    if not section:
        return "请指定要翻译的节标题，例如 section='Abstract'；整篇请按节分多次调用。"

    try:
        data = translator.translate(filename, section)
    except ValueError as e:
        # 文件名或节标题写错时，把可用值回给模型，它会自己改参数重新调用
        return "翻译失败：%s" % e

    # 模型只认字符串，所以把 {标题: {page, zh}} 拼成一段带页码的文本。
    # 页码一定要带上：用户问「这句在哪页」时模型只能从这里拿到答案
    parts = []
    for title in data:
        item = data[title]
        parts.append("【%s】第 %s 页\n%s" % (title, item["page"], item["zh"]))
    return "\n\n".join(parts)
