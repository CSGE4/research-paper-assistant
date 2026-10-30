from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from knolege import Knolege
import config
import os
import jieba
from dotenv import load_dotenv
from bm25_score import hybrid_search
load_dotenv()

# 整个进程共用一份 Knolege。这么写有两个后果，都必须知道：
#   1) import 本模块的那一刻就会把 Chroma、DashScope 客户端、bm25.pkl 全部打开，
#      所以 web.py 里注入 secrets 的那段必须放在 import vctor 之前，晚一行就拿到空 key；
#   2) Streamlit 的每个会话共用这一个对象，谁改了它的状态，别人也跟着变。
global_knowledge=Knolege()
class vctor_story_search(object):
    """检索层的对外门面：向量召回、BM25 召回、两路融合，都从这里进。"""

    def __init__(self):
        # 检索要用入库时建好的 BM25 索引，所以直接复用上面那份 singleton，
        # 不重新读一遍 bm25.pkl（6 篇论文的词表重新加载一次要好几秒）。
        self.kno=global_knowledge
        # 这里又建了一个 DashScopeEmbeddings 和一个 Chroma，和 Knolege 内部那套指向
        # 同一个持久化目录、同一个 collection。能用，但等于同一个库开了两个客户端，
        # 内存和连接都是双份——Streamlit 免费实例的 1G 内存一部分就耗在这。
        self.embedding = DashScopeEmbeddings(model="text-embedding-v4",dashscope_api_key=os.getenv("DASHSCOPE_API_KEY"))
        self.vector_story=Chroma(
            collection_name=config.collections_name,
            embedding_function=self.embedding,
            persist_directory=config.persist_directory,
        )
    def get_all(self):
        """把整库拿出来（ids / documents / metadatas），调试和统计块数时用。"""
        return self.vector_story.get()
    def get_structured_all(self):
        """入库时提取的结构化结果（标题/作者/摘要/标题层级/表/图/公式/参考文献/附录）。

        数据只在 chroma_structured.json 里，不经过向量库，
        所以这里延迟 import paper_extractor：用不上就不用把 1600 行加载进内存。
        """
        from paper_extractor import load_structured
        return load_structured()
    def get_structured_designate(self,filename):
        """取某一篇的结构化结果，没有就返回 None。"""
        return self.get_structured_all().get(filename)
    def get_designate(self,filename):
        """按文件名取该篇的全部块，用于展示或调试。"""
        # return self.vector_story.get()
        return self.vector_story.get(where={"source":filename})
    def hybrid_search_two(self,query:str,k=10):
        """第一段召回：向量 Top-k + BM25 Top-k，两路各返回各的，还没合并。

        返回值是个二元组 (向量结果, BM25结果)，两边结构不一样：
          向量结果是 [(Document, 余弦相似度), ...]
          BM25 结果是 [{"text", "bm25_score", "metadata"}, ...]
        合并发生在 bm25_score.hybrid_search 里。
        """
        vector_res=self.vector_story.similarity_search_with_score(query,k=k)
        res = []
        for doc, d in vector_res:  # 打分
            # similarity_search_with_score 给的是 L2 距离的平方，不是相似度。
            # 归一化向量的情况下 cos = 1 - d²/2，这一步把距离换回 0~1 的相似度，
            # 才能和 BM25 做加权（两者必须同量纲）。
            cos_sim = 1 - (d ** 2) / 2
            # 浮点误差可能算出 1.0000000002 或 -0.0000001，截断回 [0,1]
            cos_sim = max(min(cos_sim, 1.0), 0.0)
            res.append((doc, cos_sim))

        #BM25关键词召回
        # 库里还没有任何一篇（首次启动 / 全删光）时 BM25 是 None，直接只用向量那一路。
        # 注意这里返回的第二个元素是个字符串列表，跟正常返回的字典列表结构不同，
        # 下游 hybrid_search 会按字典去取，所以这个分支真被执行到会 TypeError。
        # 目前 6 篇论文一直在库，这个分支从没走到过，属于"没被证伪的兜底"。
        if self.kno.bm25 is None or len(self.kno.bm25_corpus) == 0:
            return res, ["bm25 is worry"]
        tokenized_query = jieba.lcut(query)
        scores=self.kno.bm25.get_scores(tokenized_query)
        score_with_index = []
        for idx, score in enumerate(scores):
            # 存 (-分数, 下标) 再升序排，等价于按分数降序；
            # 带上 idx 是为了排序稳定，分数相同时保持原顺序。
            score_with_index.append((-score, idx))  # 负号，方便从小到大=分数从大到小
        score_with_index.sort()
        result = []
        for item in score_with_index[:k]:
            idx = item[1]
            score_val = scores[idx]
            # 这行的本意是"分数太低就跳过"，但 BM25 分数恒为负？不——这里取的是
            # 原始 scores[idx]，恒 >= 0，所以 score_val <= -2 永远不成立，是句死代码。
            # 历史遗留：早期用的是负数排序，忘了把判断一起改过来。
            if score_val <= -2:
                continue
            # bm25_corpus 里存的是分词后的列表，拼回字符串才能当文本用
            chunk_text = "".join(self.kno.bm25_corpus[idx])
            meta_info = self.kno.bm25_chunk_source[idx]
            result.append({
                "text": chunk_text,
                "bm25_score": float(score_val),
                "metadata": meta_info
            })
        return res,result  # bm25_res为纯文本
    def result_search(self,query,k=10):
        """对外的检索入口：两路召回 → 融合排序 → 返回 Top-k 带 metadata 的块。

        注意先把整库文本拉出来传给融合函数，因为要重算 BM25 的绝对分；
        这一步是每次查询都做的，也是目前检索延迟的主要来源。
        """
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
    # 单独跑本文件可以验一次召回：打印全库的块 id，用来确认入库是否成功
    a=vctor_story_search()
    # print(a.get("1706.03762v1.pdf"))
    # print(a.hybrid_search_two("1706.03762v7"))
    # print(a.hybrid_search_two("重庆文理学院 杨熙 四级"))
    print(a.get_all()["ids"])
