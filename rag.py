from langchain_core.runnables import RunnablePassthrough,RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser

from vctor import vctor_story_search
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()
import os
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
def prints(s):
    print("*"*20)
    print(s.to_string())
    print("*"*20)
    print("*"*20)
    return s

class Rag:
    def __init__(self):
        self.vector_story=vctor_story_search()

        self.model=ChatOpenAI(
            model="qwen3.8-max",
            base_url=os.getenv("qwen_url"),
            api_key=os.getenv("DASHSCOPE_API_KEY"),
        )

        self.prompt=ChatPromptTemplate.from_messages(
            [
                ("system","以我提供的已知参考资料为主,简洁和专业的回答用户问题。参考资料:{context}。在回答结尾，请详细的显示文档的来源，来自哪一篇文件，第几页。如有可能，也可以显示在第几段。如果参考资料内容与问题无关，请输出空。"),
                ("user", "请回答用户提问：{input}")
            ]
        )

        self.chain=self.__get_chain_no_retriever()

    def collection_source(self,result_search):
        """输入: [{"text":..., "metadata":{"source":..., "page_number":...}}, ...]
        输出: "1706.03762v1.pdf 第3页、第7页、第11页"""
        ai = dict()
        for i in result_search:
            source = i["metadata"]["source"]
            page = i["metadata"]["page_number"]
            ai.setdefault(source, set())
            ai.get(source).add(page)
        return ai
    def __get_chain_no_retriever(self):
        chain=(
            self.prompt|prints|self.model|StrOutputParser()
        )
        return chain
    def get_answer(self,query:str):
        retriever=self.vector_story
        result=retriever.result_search(query)
        context = "\n\n".join(i["text"] for i in result)
        answer = self.chain.invoke({"input": query, "context": context})
        sources=self.collection_source(result)
        return {"answer": answer, "sources": sources}
if __name__=="__main__":
    # rag=rag().chain.invoke({"input":"我的项目有那些"})
    # print(rag)
    sa=Rag().get_answer("BLEU 分数是多少？")
    print(sa)