from rag import Rag
rag_instance=None
def local_search(query:str):
    """在本地知识库检索文档，仅查询本地资料，不会联网搜索
        Args:
            query: 用户的查询问题
        """
    global rag_instance
    if rag_instance is None:
        rag_instance=Rag()
    result=rag_instance.get_answer(query)

    # 模型只认文本，所以把 dict 拼成字符串。
    # 来源行固定成「检索文件来源以及页码：...」这个格式，
    # 提示词里要求模型原样保留的正是这一行，格式一改规则就失效了。
    sources=[]
    for name in result["sources"]:
        pages=sorted(result["sources"][name])
        page_text=""
        for p in pages:
            if page_text:
                page_text=page_text+"、"
            page_text=page_text+"第%d页" % p
        sources.append("%s %s" % (name, page_text))
    if not sources:
        sources.append("无")

    return "%s\n\n检索文件来源以及页码：%s" % (result["answer"], "；".join(sources))
if __name__=="__main__":
    a=local_search("我的四级成绩是多少")
    print(a)