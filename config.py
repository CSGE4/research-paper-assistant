# 全项目的路径与参数都集中在这里，其它模块只读不写。
# 所有路径都是相对路径，所以启动 Streamlit 时必须在项目根目录，
# 换个目录启动就会在别处新建一套空库，看起来像"数据丢了"。

md5_path="./md5.txt"                      # 已入库文件的指纹，防重复上传
bm25_cache="./bm25.pkl"                   # BM25 分词结果 + 每个块对应的文件/页码
file_path_embedding=r"./embedding.json"   # 每篇论文的页级向量，用于入库时查重
file_path_summary=r"./chroma_summary.json"  # 每篇论文的前 2000 字摘要，给相似度判断用
# 结构化提取结果（标题/作者/摘要/标题层级/表/图/公式/参考文献/附录），入库时写入
file_path_structured=r"./chroma_structured.json"
sqlite_path="./sqlite.db"                 # LangGraph 的会话记忆（按 thread_id 存对话）
CACHE_PATH = r"./translate_cache.json"    # 缓存：翻译过的段落不必再花钱
RESULT_PATH = r"./translate.json"         # 整篇翻译结果落盘，供页面重复打开时直接读

collections_name="knolege"                # Chroma 的 collection 名
persist_directory="./chroma"              # 向量库目录，已随仓库提交，线上靠它免去重新嵌入

# 递归切分器的分隔符，顺序就是优先级：从上往下依次尝试，切不动才用下一个。
# 英文论文的关键在于 ". " 后面那个空格——不带空格的话，
# "3.5"、"et al." 这类会被当成句子边界，把句子切成两半。
separators = [
    "\n\n",   # 论文段落/小节分隔符，最高优先级
    "\n",
    ". ", "! ", "? ", # 英文句子结束，注意句号后面带空格，英文标准写法
    "; ",    # 分号分隔并列长句
    ", ",    # 逗号放到后面，迫不得已才切逗号，不切断复合句
    " ",
    ""
]

# 语义切块之后的兜底尺寸：单块上限 1500 字符，相邻块重叠 150 字符。
# 重叠是为了让被切断的跨块语义在两侧各留一段，Top-10 同时召回时能拼回来。
chunk_size=1500
chunk_overlap=150

# 只给 agent.py 的 __main__ 自测用；页面上的会话 id 是 pages/app.py 里按浏览器会话生成的
config_thread={"configurable":{"thread_id":"1"}}
