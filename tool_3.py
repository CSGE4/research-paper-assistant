from rag import Rag
from tool_2 import search_tools
import json
rag_instance=None
def local_search(query:str):
    """在本地知识库检索文档，仅查询本地资料，不会联网搜索
        Args:
            query: 用户的查询问题
        """
    global rag_instance
    if rag_instance is None:
        rag_instance=Rag()
    answer=rag_instance.get_answer(query)
    return answer
if __name__=="__main__":
    a=local_search("我的四级成绩是多少")
    print(a)