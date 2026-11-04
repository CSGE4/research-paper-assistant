def list_documents():
    """列出本地文献库里有哪些文献（文件名、标题、页数）。
       不知道库里有什么、或者要覆盖多篇文献时，先调用这个再检索。
    """
    # 结构化结果里就有标题和页数，不用翻向量库，也不花模型调用，几乎瞬时返回。
    # 这个工具解决的是一个很实际的问题：模型不知道库里有什么，只能猜关键词，
    # 猜偏一次就白花十几秒。有了文件名和标题，它可以直接按标题检索。
    from paper_extractor import load_structured
    docs = load_structured()
    if not docs:
        return "本地文献库是空的，一篇文献都没有。"
    lines = []
    for name in sorted(docs):
        info = docs.get(name) or {}
        title = ((info.get("title") or {}).get("value") or "").strip()
        pages = info.get("page_count")
        page_text = ""
        if pages:
            page_text = "  %s页" % pages
        if title:
            lines.append("%s  %s%s" % (name, title, page_text))
        else:
            lines.append("%s%s" % (name, page_text))
    return "本地文献库共 %d 篇：\n%s" % (len(docs), "\n".join(lines))