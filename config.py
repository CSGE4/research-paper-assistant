md5_path="./md5.txt"
bm25_cache="./bm25.pkl"
file_path_embedding=r"./embedding.json"
file_path_summary=r"./chroma_summary.json"
# 结构化提取结果（标题/作者/摘要/标题层级/表/图/公式/参考文献/附录），入库时写入
file_path_structured=r"./chroma_structured.json"
sqlite_path="./sqlite.db"
CACHE_PATH = r"./translate_cache.json"    # 缓存：翻译过的段落不必再花钱
RESULT_PATH = r"./translate.json"

collections_name="knolege"
persist_directory="./chroma"

separators = [
    "\n\n",   # 论文段落/小节分隔符，最高优先级
    "\n",
    ". ", "! ", "? ", # 英文句子结束，注意句号后面带空格，英文标准写法
    "; ",    # 分号分隔并列长句
    ", ",    # 逗号放到后面，迫不得已才切逗号，不切断复合句
    " ",
    ""
]

chunk_size=1500
chunk_overlap=150

config_thread={"configurable":{"thread_id":"2"}}