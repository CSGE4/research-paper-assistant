from langchain_core.runnables import RunnablePassthrough,RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser

from vctor import vctor_story_search
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()
import os
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder

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
                ("system","只允许复述参考资料里明确写出的陈述。参考资料:{context}。"
           "回答结尾必须标注依据所在页码，页码只能来自上面每段开头的 [第N页]，"
           "不许自己推算。禁止从文件名、arXiv 编号、DOI、日期后缀推断任何信息"
           "（例如『1706』不代表 2017 年）。"
           "参考文献条目里的年份、作者、期刊属于被引文献，不能当成本文的信息。"
           "参考资料无法回答时，只输出「参考资料未提及」这四个字，不要补充常识。"),
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
            self.prompt|self.model|StrOutputParser()
        )
        return chain
    def get_answer(self,query:str):
        retriever=self.vector_story
        result=retriever.result_search(query)
        context = "\n\n".join("[第%s页] %s" % (i["metadata"]["page_number"], i["text"]) for i in result)
        answer = self.chain.invoke({"input": query, "context": context})
        sources=self.collection_source(result)
        return {"answer": answer, "sources": sources}
if __name__=="__main__":
    # rag=rag().chain.invoke({"input":"我的项目有那些"})
    # print(rag)
    sa=Rag().get_answer("BLEU 分数是多少？")
    print(sa)