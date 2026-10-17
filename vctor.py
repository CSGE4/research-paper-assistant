from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from knolege import Knolege
import config
import os
import jieba
from dotenv import load_dotenv
from bm25_score import hybrid_search
load_dotenv()

global_knowledge=Knolege()
class vctor_story_search(object):
    def __init__(self):
        self.kno=global_knowledge
        self.embedding = DashScopeEmbeddings(model="text-embedding-v4",dashscope_api_key=os.getenv("DASHSCOPE_API_KEY"))
        self.vector_story=Chroma(
            collection_name=config.collections_name,
            embedding_function=self.embedding,
            persist_directory=config.persist_directory,
        )
    def get_all(self):
        return self.vector_story.get()
    def get_structured_all(self):
        """入库时提取的结构化结果（标题/作者/摘要/标题层级/表/图/公式/参考文献/附录）。"""
        from paper_extractor import load_structured
        return load_structured()
    def get_structured_designate(self,filename):
        return self.get_structured_all().get(filename)
    def get_designate(self,filename):
        # return self.vector_story.get()
        return self.vector_story.get(where={"source":filename})
    def hybrid_search_two(self,query:str,k=10):
        vector_res=self.vector_story.similarity_search_with_score(query,k=k)
        res = []
        for doc, d in vector_res:  # 打分
            cos_sim = 1 - (d ** 2) / 2
            cos_sim = max(min(cos_sim, 1.0), 0.0)
            res.append((doc, cos_sim))

        #BM25关键词召回
        if self.kno.bm25 is None or len(self.kno.bm25_corpus) == 0:
            return res, ["bm25 is worry"]
        tokenized_query = jieba.lcut(query)
        scores=self.kno.bm25.get_scores(tokenized_query)
        score_with_index = []
        for idx, score in enumerate(scores):
            score_with_index.append((-score, idx))  # 负号，方便从小到大=分数从大到小
        score_with_index.sort()
        result = []
        for item in score_with_index[:k]:
            idx = item[1]
            score_val = scores[idx]
            if score_val <= -2:
                continue
            chunk_text = "".join(self.kno.bm25_corpus[idx])
            meta_info = self.kno.bm25_chunk_source[idx]
            result.append({
                "text": chunk_text,
                "bm25_score": float(score_val),
                "metadata": meta_info
            })
        return res,result  # bm25_res为纯文本
    def result_search(self,query,k=10):
        res=self.hybrid_search_two(query,k=k)
        vector_res = res[0]
        bm25_res= res[-1]
        all_doc = self.vector_story.get()
        all_doc_texts = all_doc["documents"]
        final_result = hybrid_search(
            vector_res=vector_res,
            bm25_res=bm25_res,
            all_doc_texts=all_doc_texts,
            query=query,
            k=k,
        )
        return final_result

if __name__=="__main__":
    a=vctor_story_search()
    # print(a.get("1706.03762v1.pdf"))
    # print(a.hybrid_search_two("1706.03762v7"))
    # print(a.hybrid_search_two("重庆文理学院 杨熙 四级"))
    print(a.result_search("重庆文理学院 杨熙 四级"))