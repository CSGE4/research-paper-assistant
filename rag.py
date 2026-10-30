from langchain_core.runnables import RunnablePassthrough,RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser

from vctor import vctor_story_search
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()
import os
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder

# 这一层是「检索 → 拼上下文 → 让模型照着原文答」的最小闭环。
# 它不负责对话记忆、不负责选工具，那些在 agent.py；
# 它的职责只有一件事：保证模型看到的是库里的原文，而不是它自己的常识。


class Rag:
    def __init__(self):
        self.vector_story=vctor_story_search()

        self.model=ChatOpenAI(
            model="qwen3.8-max",
            base_url=os.getenv("qwen_url"),
            api_key=os.getenv("DASHSCOPE_API_KEY"),
        )

        # 提示词里那几条限制，每一条都是针对一个真实会犯的错：
        #   ① 只允许复述原文 —— 否则模型会用自己的知识补细节，答得对但库里没有；
        #   ② 页码只能取自 [第N页] —— 否则它会自己推算页码，看起来精确其实编的；
        #   ③ 禁止从文件名推断 —— 1706.03762 是 arXiv 编号，不是 2017 年；
        #   ④ 参考文献的年份不算本文 —— 被引文献的年份极易被误当成本文发表年；
        #   ⑤ 答不了就输出固定四个字 —— 给上层一个可靠的"无结果"信号。
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

        # 名字里的 no_retriever 是有意的：这条链上没挂 retriever，
        # 检索在 get_answer 里手动做完了。因为我们要的是"混合检索"，
        # langchain 的 retriever 接口只认单一检索器，塞不进两路融合。
        self.chain=self.__get_chain_no_retriever()

    def collection_source(self,result_search):
        """输入: [{"text":..., "metadata":{"source":..., "page_number":...}}, ...]
        输出: "1706.03762v1.pdf 第3页、第7页、第11页"""
        # 同一个文件可能有多块入选，所以用 set 去重，最后得到"文件 → 页码集合"
        ai = dict()
        for i in result_search:
            source = i["metadata"]["source"]
            page = i["metadata"]["page_number"]
            ai.setdefault(source, set())
            ai.get(source).add(page)
        return ai
    def __get_chain_no_retriever(self):
        """prompt → 模型 → 取纯文本，这条链就是一次普通的问答调用。"""
        chain=(
            self.prompt|self.model|StrOutputParser()
        )
        return chain
    def get_answer(self,query:str):
        """检索 + 作答。返回 {"answer": 模型写的回答, "sources": {文件名: {页码}}}。

        关键设计：sources 是从【检索结果】里算的，不是从模型回答里解析的。
        模型自己写的页码不可信（它可能只用了 10 块里的 2 块，却把 10 块页码全写上），
        所以真正给用户看的来源，由检索层说了算。
        """
        retriever=self.vector_story
        result=retriever.result_search(query)
        # 每块前面加 [第N页]，一是让模型知道这段出自哪页可以被引用，
        # 二是让它在写"依据页码"时有可抄的东西，不用自己猜。
        context = "\n\n".join("[第%s页] %s" % (i["metadata"]["page_number"], i["text"]) for i in result)
        answer = self.chain.invoke({"input": query, "context": context})
        sources=self.collection_source(result)
        return {"answer": answer, "sources": sources}
if __name__=="__main__":
    # rag=rag().chain.invoke({"input":"我的项目有那些"})
    # print(rag)
    sa=Rag().get_answer("BLEU 分数是多少？")
    print(sa)
