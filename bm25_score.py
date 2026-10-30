import math
from collections import Counter
import jieba
"""该混合检索的权重值取 0.20，由 37 题验证集扫描得出（α 从 0.10 到 0.80）。
实测结果：α≥0.60 明显变差（MRR 0.60→0.49），0.10~0.40 区间差异在 0.02 量级的噪声内，
取 0.20 作折中——MRR 0.585，高于纯向量 0.532 和纯 BM25 0.423。
概念型和精确型查询的最优 α 是相反的——概念题要 α≈0.2~0.3，精确题要 α≈0.6
计划在后续增加动态权重选择，增加检索结果的准确性
"""

# 这个文件只干一件事：把「向量召回」和「BM25 召回」两路结果合成一个分数排序。
#
# 难点不在加权公式，在"两边分数的量纲不一样"：
#   向量那边是余弦相似度，天然落在 0~1；
#   BM25 那边是原始分，能到十几甚至几十，而且跟语料规模、文档长度都有关系。
# 直接加权等于让 BM25 单方面说了算，所以必须先把它压到 0~1 再合。


class BM25Calculator:
    """纯Python实现BM25算分器，计算真实BM25分数（不是排名分）

    为什么不用 rank_bm25 的 BM25Okapi 直接算：
      knolege.py 里那个 BM25Okapi 是拿【全部语料】建索引的，语料一变就得重建，
      而且 get_scores 返回的是"整个查询对每一篇文档"的分数向量，没法单独算某一条候选。
      这里要的是：给定查询和【一条候选块】，算出它的绝对分，才能跟向量分加权。
      所以另写一个，索引在 hybrid_search 里现建一次，只服务这一次查询。
    """

    def __init__(self, docs):
        # 全部语料分词。jieba 对英文按空格切，对中文才真正分词；
        # 语料是英文论文，所以这一层主要是把大小写、标点规整掉。
        self.docs = [[word.lower() for word in jieba.lcut(doc)] for doc in docs]  # 分词
        self.N = len(self.docs)  # 总文档数
        self.avgdl = sum(len(d) for d in self.docs) / self.N if self.N else 1  # 平均文档长度
        self.df = self._compute_df()  # 词频：多少文档包含该词
        self.idf = self._compute_idf()  # 逆文档频率（词越重要分越高）

        # BM25标准参数，这两个值是文献里的通用取值，没针对本项目调过
        self.k1 = 1.5   # 词频饱和速度：越大，同一词重复出现越"值钱"
        self.b = 0.75   # 文档长度惩罚强度：1=完全按长度归一，0=完全不看长度

    def _compute_df(self):
        """统计每个词出现在多少篇文档里（df，document frequency）。

        注意这里是 set(doc)，同一篇里出现两次只算一次——
        df 衡量的是"这个词有多常见"，不是"出现了多少次"。
        """
        df = Counter()
        for doc in self.docs:
            unique_words = set(doc)
            for word in unique_words:
                df[word] += 1
        return df

    def _compute_idf(self):
        """用标准 BM25 的 IDF 公式把 df 换算成权重。

        公式末尾的 +1 是把结果抬到正数区间：不加的话，
        出现频率超过一半文档的词会得到负分，等于"越常见的词越是减分项"。
        """
        idf = {}
        for word, df in self.df.items():
            # 标准BM25的IDF公式
            idf[word] = math.log((self.N - df + 0.5) / (df + 0.5) + 1)
        return idf

    def get_score(self, query, doc_text):
        """计算 1个查询 对 1个文档 的真实BM25分数

        这里会重新对 doc_text 分一次词，虽然 __init__ 里已经切过。
        多花这点时间换的是"能对任意传入的文本算分"，不管是语料里的还是临时拼的。
        """
        query_words = [word.lower() for word in jieba.lcut(query)]
        doc_words = [word.lower() for word in jieba.lcut(doc_text)]
        doc_len = len(doc_words)
        tf = Counter(doc_words)

        score = 0.0
        for q in query_words:
            # 查询词不在 idf 表里 = 整个语料都没出现过，对本文档的贡献就是 0
            if q not in self.idf:
                continue
            # 分子：词频经过 k1 饱和，出现再多也不会无限加分
            numerator = tf[q] * (self.k1 + 1)
            # 分母：文档越长，同样的词频得分越低（b 控制惩罚强度）
            denominator = tf[q] + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
            score += self.idf[q] * numerator / denominator
        return score


def hybrid_search(vector_res, bm25_res, all_doc_texts, query, k=8, alpha=0.20):
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
    # 代价是每次查询都要把全部块重新分词一遍，这是目前检索延迟的主要来源之一；
    # 好处是分数在同一套 idf 下可比，不会因为两次查询的语料不同而漂移。
    bm25 = BM25Calculator(all_doc_texts)

    # 2. 映射字典 key=chunk文本，存储向量分数、bm25分数、metadata溯源信息
    # 用块文本本身当 key，是因为两路召回拿到的是同一个库里的块，文本相同就一定是同一块，
    # 不需要再去对齐 id；代价是文本完全重复的块会被合并成一条。
    score_map = defaultdict(lambda: {"vec": 0.0, "bm25": 0.0, "metadata": {}})

    # 3. 录入向量检索结果：vec分数 + metadata
    # 只被向量一路召回的块，bm25 分保持默认 0.0，不会被丢掉——
    # 这是"两路取并集"而不是"取交集"的关键。
    for doc_item in vector_res:
        doc, vec_score = doc_item
        text = doc.page_content
        score_map[text]["vec"] = vec_score
        # 向量Document自带元数据，存入溯源
        score_map[text]["metadata"] = doc.metadata

    # 4. 录入BM25结果：重新计算bm25分数，同时保存bm25对应的metadata溯源
    for bm_item in bm25_res:
        text = bm_item["text"]
        # 重新计算bm25分数，不用 bm_item["bm25_score"]
        bm25_score = bm25.get_score(query, text)
        score_map[text]["bm25"] = bm25_score
        # 如果当前文本还没有metadata，就使用bm25这边的meta信息
        if not score_map[text]["metadata"]:
            score_map[text]["metadata"] = bm_item["metadata"]

    contents = list(score_map.keys())
    vec_scores = [score_map[c]["vec"] for c in contents]
    bm25_scores_raw = [score_map[c]["bm25"] for c in contents]

    # 5. BM25归一化函数
    # 用 sigmoid 而不是 min-max：min-max 会把本次的最高分强行拉成 1，
    # 于是"全都很差"和"有一篇很好"看起来一模一样，融合就失去意义。
    # 这里的做法是给一个绝对锚点——原始分到 4.0 附近算"有相关"，减 beta 之后
    # 除以 scale 压进 sigmoid，分数就能跨查询比较。
    def bm25_abs_normalize(score, beta=4.0, scale=4.0, raw_threshold=0.5):
        if score < raw_threshold:
            return 0.0
        x = (score - beta) / scale
        return 1.0 / (1.0 + math.exp(-x))

    n_bm25 = [bm25_abs_normalize(s) for s in bm25_scores_raw]
    n_vec = vec_scores

    # 6. 加权计算总分，打包全部信息
    # 两路原始分也一起带出去，调试用：结果看起来怪的时候，
    # 能直接看出是哪一路把分数顶上来的。
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
    # 单独跑本文件可以完整验一次检索，输出里带两路原始分，方便判断融合是否异常
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
