from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()
import os
from vctor import vctor_story_search
from paper_extractor import PaperExtractor
class tree:
    def __init__(self):
        self.vector=vctor_story_search()
        # 复用同一个向量库连接，不重复建 Chroma
        self.extractor=PaperExtractor(vector=self.vector)
        # self.extractor.get_all()      -> 全部切块的结构化信息
        # self.extractor.get_designate("xxx.pdf") -> 指定文件名

        self.model=ChatOpenAI(
            model="qwen3.8-max",
            base_url=os.getenv("qwen_url"),
            api_key=os.getenv("DASHSCOPE_API_KEY")
        )

        self.promote="""任务：从下面文本中提取实体以及实体之间的关系。
                规则：
                1. 只输出JSON，禁止输出任何说明、注释、markdown、```标记。
                2. nodes：实体列表，仅保留关键名词实体。
                3. edges：实体之间的关系，source/target填写实体名称。
                输出格式严格如下，不要增加额外字段：
                {"nodes": [{"name":"实体名称"}], "edges": [{"source":"实体A","target":"实体B","relation":"二者的关系"}]}
                
                文本内容：{text}
                """
    def get_all_docs(self):
        return self.vector.get_all()
    def get_designate_docs(self,filename):
        return self.vector.get_designate(filename)
    # 入库时落盘的结构化结果（完整内容提取，准且快；无需重新解析切块）
    def get_structured_all(self):
        return self.vector.get_structured_all()
    def get_structured_designate(self,filename):
        return self.vector.get_structured_designate(filename)
    # ---- 结构化信息提取（标题/作者/摘要/标题层级/表/图/公式/参考文献/附录）----
    def extract_all(self):
        return self.extractor.get_all()
    def extract_designate(self,filename):
        return self.extractor.get_designate(filename)
    def extract_summary(self,filename=None):
        return self.extractor.summary(filename)
    def get_chain(self):
        pass
    def sasd(self):
        all_docs=self.get_all_docs()
        for i in range(len(all_docs["ids"])):
            content=dict()
            id=all_docs["ids"][i]
            doc=all_docs["documents"][i]
            metadata=all_docs["metadatas"][i]
            content["id"]=id
            content["doc"]=doc
            content["metadata"]=metadata
            # print(content)

if __name__=="__main__":
    import json
    t=tree()
    # 1) 原始切块
    # print(t.get_all_docs())
    # print(t.get_designate_docs("四级.pdf"))
    # 2) 全部文件的结构化信息（标题/作者/摘要/标题层级/表格/图/公式/参考文献/附录/章节）
    all_info=t.extract_all()
    print(json.dumps(t.extract_summary(), ensure_ascii=False, indent=2))
    # print(json.dumps(all_info, ensure_ascii=False, indent=2, default=str))
    # 3) 指定文件
    # one=t.extract_designate("四级.pdf")
    # print(json.dumps(one, ensure_ascii=False, indent=2, default=str))
    # 4) 只取某一类
    # print(t.extractor.get_table("四级.pdf"))
    # print(t.extractor.get_heading("四级.pdf"))
    # print(t.extractor.get_formula("四级.pdf"))