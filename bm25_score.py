import math
from collections import Counter
import jieba
"""该混合检索的权重值修改为0.35是综合最优结果，通过ai演算得出的。
概念型和精确型查询的最优 α 是相反的——概念题要 α≈0.2~0.3，精确题要 α≈0.6
计划在后续增加动态权重选择，增加检索结果的准确性
"""
class BM25Calculator:
    """纯Python实现BM25算分器，计算真实BM25分数（不是排名分）"""

    def __init__(self, docs):
        self.docs = [[word.lower() for word in jieba.lcut(doc)] for doc in docs]  # 分词
        self.N = len(self.docs)  # 总文档数
        self.avgdl = sum(len(d) for d in self.docs) / self.N if self.N else 1  # 平均文档长度
        self.df = self._compute_df()  # 词频：多少文档包含该词
        self.idf = self._compute_idf()  # 逆文档频率（词越重要分越高）

        # BM25标准参数
        self.k1 = 1.5
        self.b = 0.75

    def _compute_df(self):
        df = Counter()
        for doc in self.docs:
            unique_words = set(doc)
            for word in unique_words:
                df[word] += 1
        return df

    def _compute_idf(self):
        idf = {}
        for word, df in self.df.items():
            # 标准BM25的IDF公式
            idf[word] = math.log((self.N - df + 0.5) / (df + 0.5) + 1)
        return idf

    def get_score(self, query, doc_text):
        """计算 1个查询 对 1个文档 的真实BM25分数"""
        query_words = [word.lower() for word in jieba.lcut(query)]
        doc_words = [word.lower() for word in jieba.lcut(doc_text)]
        doc_len = len(doc_words)
        tf = Counter(doc_words)

        score = 0.0
        for q in query_words:
            if q not in self.idf:
                continue
            numerator = tf[q] * (self.k1 + 1)
            denominator = tf[q] + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
            score += self.idf[q] * numerator / denominator
        return score


def hybrid_search(vector_res, bm25_res, all_doc_texts, query, k=8, alpha=0.35):
    """
    适配真实数据的混合检索：向量检索 + BM25精准加权
    :param vector_res: 向量结果：列表 -> [(Document, score), ...]
    :param bm25_res: BM25结果：列表 -> [{"text": str, "bm25_score": float, "metadata": dict}, ...]
    :param all_doc_texts: 所有文档原文（用于BM25算分）
    :param query: 用户查询问题
    :param k: 返回topk条结果
    :param alpha: BM25权重（0~1，越大越依赖关键词）
    :return: list[dict] 排序后的结果，包含text、分数、metadata溯源
    """
    from collections import defaultdict
    import math

    # 1. 初始化BM25算分器（你要求：重新计算bm25分数，不直接复用bm25_res里的bm25_score）
    bm25 = BM25Calculator(all_doc_texts)

    # 2. 映射字典 key=chunk文本，存储向量分数、bm25分数、metadata溯源信息
    score_map = defaultdict(lambda: {"vec": 0.0, "bm25": 0.0, "metadata": {}})

    # 3. 录入向量检索结果：vec分数 + metadata
    for doc_item in vector_res:
        doc, vec_score = doc_item
        text = doc.page_content
        score_map[text]["vec"] = vec_score
        # 向量Document自带元数据，存入溯源
        score_map[text]["metadata"] = doc.metadata

    # 4. 录入BM25结果：重新计算bm25分数，同时保存bm25对应的metadata溯源
    for bm_item in bm25_res:
        text = bm_item["text"]
        # 重新计算bm25分数，不用bm_item["bm25_score"]
        bm25_score = bm25.get_score(query, text)
        score_map[text]["bm25"] = bm25_score
        # 如果当前文本还没有metadata，就使用bm25这边的meta信息
        if not score_map[text]["metadata"]:
            score_map[text]["metadata"] = bm_item["metadata"]

    contents = list(score_map.keys())
    vec_scores = [score_map[c]["vec"] for c in contents]
    bm25_scores_raw = [score_map[c]["bm25"] for c in contents]

    # 5. BM25归一化函数
    def bm25_abs_normalize(score, beta=4.0, scale=4.0, raw_threshold=0.5):
        if score < raw_threshold:
            return 0.0
        x = (score - beta) / scale
        return 1.0 / (1.0 + math.exp(-x))

    n_bm25 = [bm25_abs_normalize(s) for s in bm25_scores_raw]
    n_vec = vec_scores

    # 6. 加权计算总分，打包全部信息
    score_list = []
    for idx, text in enumerate(contents):
        final_score = alpha * n_bm25[idx] + (1 - alpha) * n_vec[idx]
        vec_score_raw = vec_scores[idx]
        bm25_raw = bm25_scores_raw[idx]
        meta_data = score_map[text]["metadata"]
        score_list.append({
            "text": text,
            "final_score": final_score,
            "vec_score": vec_score_raw,
            "bm25_score": bm25_raw,
            "metadata": meta_data
        })

    # 7. 降序排序，取top‑k
    score_list.sort(key=lambda x: x["final_score"], reverse=True)
    topk_result = score_list[:k]

    return topk_result


if __name__ == '__main__':
    from vctor import vctor_story_search
    vector_shi_li= vctor_story_search()
    query = "BLEU 分数是多少？"
    res=vector_shi_li.hybrid_search_two(query)
    # print(len(res))
    vector_res=res[0]
    bm25_res =res[-1]
    # print(vector_res,"\n", bm25_res)
    # print("\n")
    # all_doc_texts = [你的全部文档文本列表]
    all_doc=vector_shi_li.vector_story.get()
    # print(all_doc["documents"])
    all_doc_texts=all_doc["documents"]
    # # 混合检索
    # final_result_score = finall_score(
    #     vector_res=vector_res,
    #     bm25_res=bm25_res,
    #     all_doc_texts=all_doc_texts,
    #     query=query,
    # )
    # #
    # # # 查看结果
    # print(final_result_score)
    # for doc in final_result:
    #     print(doc.page_content)
    #     print("123\n")
    doc_texts=hybrid_search(
        vector_res=vector_res,
    bm25_res=bm25_res,
    all_doc_texts=all_doc_texts,
    query=query,)
    print(doc_texts)