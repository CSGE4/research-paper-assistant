import pymupdf
import pymupdf4llm
from langchain_chroma import Chroma
import os
import base64
import datetime
from rank_bm25 import BM25Okapi
import pickle
import jieba
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
from md5_saved import *
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import DashScopeEmbeddings
load_dotenv()

# 本文件是「入库链」的全部实现：一份 PDF 进来，最后写成六个地方。
#
#   1. chroma/                 向量库，主键是"文件名_p页码_块号"
#   2. bm25.pkl                BM25 的分词结果，以及每个块属于哪个文件哪一页
#   3. embedding.json          每篇的页级向量，用来判断"这篇是不是已经传过/很像"
#   4. chroma_summary.json     每篇前 2000 字摘要，相似度判断时给模型看的材料
#   5. chroma_structured.json  规则抽取的整篇结构（标题/作者/表/公式/参考文献…）
#   6. md5.txt                 "哪些文件入过库"的账本
#
# 六处必须尽量同步，所以写入顺序是刻意排过的：见 _save_to_all_db 结尾那段注释。


class Knolege:
    def __init__(self):
        # 顺序有讲究：先把 BM25 索引读进来（load_cache），
        # 再建 embedding 客户端和 Chroma——因为 bm25.pkl 只是本地文件，最快，
        # 而 Chroma 打开要扫目录，放后面可以让前面的失败更早暴露。
        self.bm25=None
        self.bm25_corpus=[]
        self.bm25_chunk_source=[]
        self.load_cache()
        self.embeddings = DashScopeEmbeddings(
            model="text-embedding-v4",
            dashscope_api_key=os.getenv("DASHSCOPE_API_KEY")
        )
        self.chroma=Chroma(
            collection_name=config.collections_name,
            embedding_function=self.embeddings,
            persist_directory=config.persist_directory,
        )
        # 语义切块要靠向量算"相邻句子之间差多远"来决定在哪断，
        # 所以 SemanticChunker 必须拿到同一个 embeddings，不能随便换模型，
        # 换了之后切块边界会变，历史数据就对不上了。
        self.seman_splitter=SemanticChunker(
            embeddings=self.embeddings,
            breakpoint_threshold_type="percentile",  # 按百分位找断点，比固定阈值稳
            breakpoint_threshold_amount=82,          # 只在最不像的 18% 处切开
            min_chunk_size=120                       # 太碎的块不切，留给后面合并
        )
        # 语义切块的产物没有长度上限，超长的块要靠这个递归切分器兜底，
        # 分隔符优先级在 config.py 里，核心是 ". " 后面那个空格不能少。
        self.rec_splitter=RecursiveCharacterTextSplitter(
            separators=config.separators,
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            length_function=len
        )
        self.embeddings_dict={}
        self._load_embedding_store()
        self.summary_dict={}
        self._load_summary()
        # 相似文档进来时，先把解析结果缓在这，等用户选完"覆盖/都留/取消"再决定要不要入库。
        # 不这么做的话，用户点按钮时页面已经重跑，那份解析结果就没了，得重新解析一遍。
        self.temp_pdf={}

    def load_cache(self):
        """启动时把 bm25.pkl 读回来，并且把 BM25Okapi 索引现建一次。

        pkl 里只存分词结果和每块的来源信息，不存 BM25Okapi 对象本身——
        那个对象带引用，换 Python 版本可能读不出来；分词结果是纯数据，一直能读。
        """
        if os.path.exists(config.bm25_cache):
            with open(config.bm25_cache, "rb") as f:
                data=pickle.load(f)
                self.bm25_corpus=data["tokens"]
                self.bm25_chunk_source=data["source"]
                self.bm25 = BM25Okapi(self.bm25_corpus)
        else:
            # 首次启动，初始化空
            self.bm25 = None
            self.bm25_corpus = []
            self.bm25_chunk_source = []
    def build_bm25(self,new_texts,filename,page_nums):
        """把新入库的块追加进 BM25，然后整份重建索引。

        rank_bm25 不支持增量添加，所以只能"老语料 + 新语料"一起重建。
        6 篇论文、727 个块重建一次是毫秒级，等语料上到几千块就得考虑换方案。
        page_nums 和 new_texts 必须等长且一一对应，否则页码会错位。
        """
        new_tokens = [jieba.lcut(i) for i in new_texts]
        new_meta = [
            {"source": filename, "page_number": page_nums[idx]}
            for idx in range(len(new_texts))
        ]
        self.bm25_chunk_source.extend(new_meta)
        self.bm25_corpus.extend(new_tokens)
        self.bm25 = BM25Okapi(self.bm25_corpus)
        with open(config.bm25_cache, "wb") as f:
            pickle.dump({"tokens": self.bm25_corpus,"source":self.bm25_chunk_source}, f)

    def _load_summary(self):
        """从json加载历史文档向量，仅在__init__执行"""
        # 摘要给两处用：相似度判断时喂给模型，以及删除时确认这篇确实入过库。
        emb_path = config.file_path_summary
        if os.path.exists(emb_path):
            try:
                with open(emb_path, "r", encoding="utf-8") as f:
                    self.summary_dict= json.load(f)
                print(f"成功加载历史embedding记录，共 {len(self.summary_dict)} 份文档")
            except Exception as e:
                # 文件坏了就当空的，不要让一个坏 json 把整个应用卡死；
                # 代价是重复上传的检测会失效，所以这里必须打印，不能静默。
                print(f"embedding.json读取失败，重置为空: {e}")
                self.summary_dict= {}
        else:
            self.summary_dict= {}

    def _load_embedding_store(self):
        """从json加载历史文档向量，仅在__init__执行"""
        # 存的是"每篇论文 → 每一页的 embedding"，结构是 {文件名: [[...], ...]}。
        # 查重时把页向量取平均，得到整篇的代表向量，再跟历史逐篇算余弦。
        emb_path = config.file_path_embedding
        if os.path.exists(emb_path):
            try:
                with open(emb_path, "r", encoding="utf-8") as f:
                    self.embeddings_dict = json.load(f)
                print(f"成功加载历史embedding记录，共 {len(self.embeddings_dict)} 份文档")
            except Exception as e:
                print(f"embedding.json读取失败，重置为空: {e}")
                self.embeddings_dict = {}
        else:
            self.embeddings_dict = {}

    def _save_embedding_store(self):
        """持久化保存embedding列表"""
        emb_path = config.file_path_embedding
        with open(emb_path, "w", encoding="utf-8") as f:
            json.dump(self.embeddings_dict, f, ensure_ascii=False)
    def _merge_small_chunks(self,chunks,min_size=800,max_size=1500,drop_below=20):
        """把语义切块器输出的碎块合并到 min_size 以上。

        - 语义边界已由 SemanticChunker 找好，这里只在【相邻】块之间合并，不破坏边界
        - drop_below: 小于该长度的块直接丢弃（PDF 页码/页眉页脚，如 '2'、'10 \n\n'）
        - 尾巴并回上一个块，避免末尾留下一个孤立碎块
        - 合并后超过 max_size 就停止吸收
        """
        # 先丢掉页码这类碎片。PDF 每页的页眉页脚会被解析成独立的短块，
        # 它们没有任何语义，留着只会污染检索结果。
        chunks=[c for c in chunks if len(c.strip())>=drop_below]
        merged=[]
        buf=""
        for chunk in chunks:
            buf+=chunk
            if len(buf)>=min_size:
                merged.append(buf)
                buf=""
        if buf:
            # 收尾：剩下的一小截不能单独成块，优先并进上一块
            if not merged:
                merged.append(buf)      # 整页都很短：不能丢，单独成块
            elif len(merged[-1])+len(buf)<=max_size:
                merged[-1]+=buf
            else:
                merged.append(buf)
        return merged
    def get_final_chunks(self,text):
        """一块文本 → 最终入向量库的若干块，三档依次处理：

        1. SemanticChunker 按语义断点切（切在"意思变了"的地方）
        2. _merge_small_chunks 把碎块合并、把页码碎片丢掉
        3. 仍然超过 chunk_size 的，再交给 RecursiveCharacterTextSplitter

        第 3 档不能省：语义切块器只保证"切在语义边界"，不保证长度上限，
        一篇结构松散的论文可能切出 4000 字的长块，直接把 embedding 的上下文占满。
        """
        semantic_chunks=self.seman_splitter.split_text(text)
        semantic_chunks=self._merge_small_chunks(semantic_chunks)
        final_chunks=[]
        for chunk in semantic_chunks:
            if len(chunk) > config.chunk_size:
                rec_chunks=self.rec_splitter.split_text(chunk)
                final_chunks.extend(rec_chunks)
            else:
                final_chunks.append(chunk)
        return final_chunks
    def check_pdf_similarity_and_save(self,new_file_name,page_texts, page_embedding,threshold=0.8):
        """
    文档相似度校验：
    输入：
        page_texts: list[str]，PDF每页文本
        page_embedding: list[list[float]]，每页原始embedding（DashScope返回原生list）
        threshold: 相似度阈值
    返回：
        (has_similar: bool, max_sim: float, old_file_name)
        has_similar=True 代表找到相似文档，需要交给agent二次判断
        has_similar=False 代表无相似文档，可以直接入库
    """
        # 空文本保护：扫描件 / 图片型 PDF 提取不到文字。
        # 这里绝对不能返回 True —— True 的意思是"存在相似文档"，会把流程带进相似度分支，
        # 去读一个还不存在的 chroma_summary.json，最后报出"读取旧文档摘要失败"这种完全误导人的错。
        if not page_texts:
            raise Exception("这篇 PDF 没有提取到文字（可能是扫描件或图片型），请换一个有文字层的版本")
        empty_pages = 0
        for t in page_texts:
            if len(t.strip()) == 0:
                empty_pages = empty_pages + 1
        if empty_pages == len(page_texts):
            raise Exception("这篇 PDF 没有提取到文字（可能是扫描件或图片型），请换一个有文字层的版本")

        # 当前文档均值向量，用于相似度比对
        # 为什么取平均而不是逐页比：同一篇论文重传时页码可能因为重新排版而错位，
        # 逐页对齐会很脆；取整篇均值鲁棒得多，代价是"只有一章重复"检测不出来。
        doc_embed = np.mean(page_embedding, axis=0).reshape(1, -1)
        max_sim = 0.0
        file_name=None

        # 遍历历史文档，逐个计算均值向量相似度
        for old_file_name,old_file_embeds in self.embeddings_dict.items():
            # 自己和自己比必然是 1.0。重传同一篇（比如上一次入库中途失败了）时，
            # 不能把"上次留下的自己"当成"相似的另一篇"，否则会被拦在相似度分支里，
            # 然后去读一个根本没写成的摘要文件，报出"读取旧文档摘要失败"。
            if old_file_name == new_file_name:
                continue
            old_doc_embedding = np.mean(old_file_embeds, axis=0).reshape(1, -1)
            sim = cosine_similarity(doc_embed, old_doc_embedding)[0][0]
            if sim > max_sim:
                max_sim = sim
                file_name = old_file_name

        if max_sim >= threshold:
            print(f"True，存在高相似度文档，相似度：{max_sim:.4f}")
            return True, max_sim,file_name#(是否存在相似文档，最大相似度，旧的文件名)
        else:
            print(f"False，不存在高相似度文档，相似度：{max_sim:.4f}")
            return False, max_sim,None#(是否存在相似文档，最大相似度，旧的文件名)

    def simple_extract_summary(self,text,max_len=2000):
        """取前 2000 字当摘要。

        不做任何总结，就是截断——因为这份摘要的用途是"给模型看个开头，
        判断两篇是不是同一篇"，不是给用户读的，用模型总结反而慢且可能编。
        """
        return text[:max_len].replace("\n","")

    def _save_structured_info(self, filename, data):
        """入库时用完整分页内容提取结构化信息并落盘。

        两步：
          1) 规则解析（快、免费、确定）
          2) 模型兜底：只对规则不可靠的字段（标题/作者/关键词）调用大模型，
             并用"值必须能在原文核对到"来防编造。
        提取只是附加信息，任何异常都不应影响入库，因此整体 try/except 兜住。
        落盘文件是唯一数据源，这里不再额外维护内存缓存。
        """
        try:
            from paper_extractor import extract_from_pages, upsert_structured
            # 传进去的是按页解析的 data，不是切好的块：
            # 块是被切断的，跨页的表格、跨块的参考文献在这里都会丢。
            info = extract_from_pages(data, filename)
            try:
                from model_extractor import enrich
                info = enrich(info)
            except Exception as e:
                # 模型补全失败就用规则结果，不影响入库
                print(f"[knolege] 模型补全跳过（保留规则结果）{filename}: {e}")
            upsert_structured(filename, info)
            return info
        except Exception as e:
            print(f"[knolege] 结构化提取失败（不影响入库）{filename}: {e}")
            return None

    def _drop_structured_info(self, filename):
        """删文档时同步清理结构化结果，避免脏数据。"""
        try:
            from paper_extractor import drop_structured
            return drop_structured(filename)
        except Exception as e:
            print(f"[knolege] 结构化结果清理失败 {filename}: {e}")
            return None

    def _parse_pdf_base64(self, base64_str: str):
        """私有方法：只做pdf解析，返回data，只写这一处解析逻辑

        page_chunks=True 是关键：它让结果变成"每页一个 dict"，而不是整篇拼成一个字符串。
        只有这样后面才能知道每个块属于第几页，页码锚定（回答必须标页码）才有数据可依。
        """
        a = base64.b64decode(base64_str.encode("utf-8"))
        # 直接从内存里的字节流打开，不落临时文件——Streamlit Cloud 的临时目录不保证可写
        file_pdf_doc = pymupdf.open("pdf", a)
        data: list[dict] = pymupdf4llm.to_markdown(file_pdf_doc, page_chunks=True)
        file_pdf_doc.close()
        return data

    def _save_to_all_db(self,filename:str,data,page_embedding,str_hashlib):
        """公共入库：Chroma、摘要、md5、embeddings_dict、bm25统一写入"""
        metadata_list = []
        all_chunks = []
        ids = []
        page_nums=[]
        for i in data:
            data_text = i["text"]
            data_no = i["metadata"]["page_number"]
            text_chunks = self.get_final_chunks(data_text)
            all_chunks.extend(text_chunks)
            page_nums.extend([data_no]*len(text_chunks))
            for chunk_idx, _ in enumerate(text_chunks):
                metadata_list.append({
                    "source": filename,
                    "create_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "page_number": data_no,
                })
                # id 用"文件名_第几页_第几块"拼出来，是【确定性】的。
                # 好处：同一篇重传时 add_texts 会覆盖同名 id，而不是插进第二份；
                # 代价：如果重传后切出的块数变少，多出来的旧 id 不会被删掉，会残留。
                #      这时要靠 delete_file 先删干净再入库。
                ids.append(f"{filename}_p{data_no}_{chunk_idx}")
        self.chroma.add_texts(all_chunks, metadata_list, ids)
        # 对新入库的文档进行摘要，取前1200个字符为摘要
        summary = self.simple_extract_summary(data[0]["text"])
        # 将摘要增加近self.summary_dict中
        self.summary_dict[filename] = summary
        # 将self.summary_dict写入chroma_summary.json文件
        with open(config.file_path_summary, "w") as f:
            json.dump(self.summary_dict, f)
        # 结构化提取（标题/作者/摘要/标题层级/表/图/公式/参考文献/附录）
        # 此时 data 是完整的按页解析结果，页序正确、内容完整，比事后从切块反推更准
        self._save_structured_info(filename, data)
        # 将分词加入bm25模型中
        self.build_bm25(all_chunks,filename,page_nums)
        # 所有"这篇已经入库了"的登记放到最后，中途任何一步失败，库里都不会留下半条记录。
        # 否则会出现 md5.txt 拦不住重传、embedding.json 却认得"自己"的幽灵记录，
        # 重传时被误判成"和另一篇文档相似"，然后去读一个没写成的摘要文件。
        save_md5(str_hashlib,filename)
        self.embeddings_dict[filename] = page_embedding
        self._save_embedding_store()
        return True, None, filename  # (入库是否成功，旧的文件名，新的文件名)
    def upload_str_pdf(self,base64_str:str,filename:str):
        """上传入口：从 base64 到"决定这篇该怎么处理"，中间不做任何入库写入。

        返回值有三种形态，页面（web.py）按类型分支，这是本函数最容易被误读的地方：
          (False, 原名, 原名) → 内容完全重复（md5 命中），什么都没做
          (True,  None, 新名) → 无相似文档，已经入库完成
          (dict, 旧名, 新名) → 找到了相似文档，dict 是智能体给的建议，等用户选

        注意第三种情况【还没有入库】：解析结果暂存在 self.temp_pdf 里，
        用户点"覆盖/都留"之后才走 real_save_pdf 真正写库。
        """
        str_hashlib=get_str_md5(base64_str)
        file_exist()

        # 第一道闸：内容完全一样，直接返回，连解析都省了
        if check_md5(str_hashlib):
            return False,filename,filename#(入库是否成功，旧的文件名，新的文件名)

        data = self._parse_pdf_base64(base64_str)
        #拼接多页文档内容，对所有文档进行向量转化
        page_texts=[]
        for i in data:
            pages=i["text"]
            page_texts.append(pages)
        # 这里对整个文档（逐页）做一次 embedding，有两个用途：
        # 一是下面查相似文档，二是入库后存进 embedding.json 作为这篇的指纹。
        page_embedding=self.embeddings.embed_documents(page_texts)
        #解析
        # 先暂存，等确定要入库时再取用——避免用户点了"取消"却已经写了一半
        self.temp_pdf[filename]={
            "filename":filename,
            "md5":str_hashlib,
            "data":data,
            "file_embedding":page_embedding
        }
        #检查是否存在相似文档
        is_skip,max_sim,old_file_name=self.check_pdf_similarity_and_save(
            filename,page_texts, page_embedding)
        #如果存在相似文档，走下面流程,没有进行文件入库
        if is_skip:
            # 存在相似文档，调用Agent做二次决策
            if not old_file_name:
                raise Exception("相似度判断异常：没有匹配到具体文档，无法走相似度分支")
            # 延迟 import：只有真的撞上相似文档时才需要 LangChain 那一整套，
            # 平时上传不加载 agent，省启动时间
            from agent import agent
            try:
                with open(config.file_path_summary,"r",encoding="utf-8") as f:
                    all_doc=json.load(f)
                    old_file_summary=all_doc[old_file_name]
            except FileNotFoundError:
                # 摘要文件还没生成，说明库里其实没有一篇完整的旧文档，
                # 多半是上一次入库中途失败了，只留下半条记录
                raise Exception(
                    "库里还没有 chroma_summary.json，说明上一次入库没有完成。"
                    "请删掉 embedding.json、md5.txt 和 chroma/ 目录后重新上传")
            except KeyError:
                raise Exception(
                    "摘要文件里没有「%s」的记录，说明上一次入库没有完成。"
                    "请删掉 embedding.json、md5.txt 和 chroma/ 目录后重新上传" % old_file_name)
            except json.JSONDecodeError as e:
                raise Exception(f"读取旧文档摘要失败（摘要文件内容损坏）: {e}")
            new_file_summary=self.simple_extract_summary(data[0]["text"])
            # 把两份摘要和相似度数值一起交给模型，让它给出"覆盖/都留/拒绝"三选一的建议。
            # 明确要求只输出 JSON，是为了下面 parse_agent_json 能稳定解析。
            content=f"""新文档文件名为：{filename},文件摘要为：{new_file_summary}.
                        旧文档文件名为：{old_file_name},文档摘要为：{old_file_summary}.
                        对比两份文档，请给出建议：应该怎么保存文档？是否需要删除旧文档？
                        两份文档相似度为{max_sim:.4f}
                        请严格输出JSON，只能包含下面3个字段，不要额外解释：
                        1. sim_desc:相似度程度简要判断
                        2. action:只能选["覆盖旧文档入库", "两篇全部保留入库", "拒绝本次上传"]
                        3. reason:简短理由"""
            result=agent.invoke({"messages":[{"role":"user","content":content}]})
            return result,old_file_name,filename
        #如果不存在相似文档就进行切块，向量入库
        result,old_file_name,new_file_name=self._save_to_all_db(filename,data,page_embedding,str_hashlib)
        return result,old_file_name,new_file_name#(入库是否成功，旧的文件名，新的文件名)

    def real_save_pdf(self,filename:str,base64_str:str=None):#(在高相似度的情况下任然决定文档入库)
        """用户已经决定"覆盖"或"都留"之后，才真正写库。

        优先用 temp_pdf 里那份解析结果，省一次解析和一次 embedding。
        但 temp_pdf 是内存里的东西：页面重跑、会话重建、云端休眠唤醒都可能让它消失，
        所以它只是"能省就省"的快路径，不是前提——拿不到就用 base64 重新解析一遍。

        以前末尾直接 del，缓存一旦已经没了就抛 KeyError，用户点按钮当场崩，
        界面上留着一排按钮，看着像"点了没反应"，实际是已经把解析结果弄丢了。
        """
        cache=self.temp_pdf.pop(filename,None)
        if cache is not None:
            data=cache["data"]
            filename=cache["filename"]
            md5_str=cache["md5"]
            file_embedding=cache["file_embedding"]
        else:
            # 快路径失效，退回慢路径：重新解析 + 重新嵌入，多花 30~90 秒，但不会丢
            if not base64_str:
                return None,None,filename
            md5_str=get_str_md5(base64_str)
            data=self._parse_pdf_base64(base64_str)
            page_texts=[]
            for i in data:
                page_texts.append(i["text"])
            file_embedding=self.embeddings.embed_documents(page_texts)
        return self._save_to_all_db(filename,data,file_embedding,md5_str)

    def delete_file(self,filename:str):
        """删除embedding文件，summary文件，向量数据库，bm25模型,md5文件中的所属内容"""
        # 六处存储必须一起删，漏一处就会产生"删不掉的幽灵"：
        # 比如只删了 Chroma 没删 md5.txt，用户再传同一篇会被 md5 拦下，
        # 但库里其实一片都没有——表现就是"传了但搜不到"。
        #1.embedding文件
        val_embedding = None
        if os.path.exists(config.file_path_embedding):
            with open(config.file_path_embedding,"r",encoding="utf-8") as f:
                data_embedding=json.load(f)
            val_embedding=data_embedding.pop(filename,None)
            with open(config.file_path_embedding,"w",encoding="utf-8") as f:
                json.dump(data_embedding,f,ensure_ascii=False,indent=2)
        self._load_embedding_store()
        #2.summary文件
        val_summary = None
        if os.path.exists(config.file_path_summary):
            with open(config.file_path_summary,"r",encoding="utf-8") as f:
                data_summary=json.load(f)
            val_summary=data_summary.pop(filename,None)
            with open(config.file_path_summary,"w",encoding="utf-8") as f:
                json.dump(data_summary,f,ensure_ascii=False,indent=2)
        self._load_summary()
        #3.向量数据库chroma
        # 按 metadata.source 删，能一次把这篇的所有页、所有块清干净
        old_chroma_data=self.chroma.get(where={"source":filename})
        self.chroma.delete(where={"source":filename})
        #4.bm25模型
        # BM25 那两份列表是并行的（第 i 个 token 列表对应第 i 个来源），
        # 所以必须同时按同一个条件过滤，只删一边会导致页码全部错位。
        new_corpus = []
        new_source = []
        for token, src in zip(self.bm25_corpus, self.bm25_chunk_source):
            # src 是 {"source":名字, "page_number":页码} 这个字典，
            # 要取 "source" 才能跟文件名字符串比较。
            # 早期这里直接写成 src != filename，等于拿字典跟字符串比，
            # 永远不相等，一条都删不掉——删完文档后 BM25 还能召回它的块。
            if src["source"] != filename:
                new_corpus.append(token)
                new_source.append(src)
        self.bm25_corpus = new_corpus
        self.bm25_chunk_source = new_source
        if len(self.bm25_corpus) > 0:
            self.bm25 = BM25Okapi(self.bm25_corpus)
        else:
            self.bm25 = None
        with open(config.bm25_cache, "wb") as f:
            pickle.dump({"tokens": self.bm25_corpus,"source":self.bm25_chunk_source}, f)
        #5.md5文件
        remove_md5_by_filename(filename)
        #6.结构化结果文件
        self._drop_structured_info(filename)
        return val_embedding,val_summary,old_chroma_data,self.chroma.get(where={"source":filename})

if __name__=="__main__":
    # 单独跑本文件 = 命令行入库一篇论文，用来验证切块、抽取、写库整条链
    import pymupdf4llm
    service=Knolege()
    file=r"C:\Users\34078\Desktop\1706.03762v1.pdf"
    filename="1706.03762v1.pdf"
    with open(file,"rb") as f:
        data=f.read()
    base64_str = base64.b64encode(data).decode("utf-8")
    a=service.upload_str_pdf(base64_str,filename)
    print(a)
