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

class Knolege:
    def __init__(self):
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
        self.seman_splitter=SemanticChunker(
            embeddings=self.embeddings,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=82,
            min_chunk_size=120
        )
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
        self.temp_pdf={}

    def load_cache(self):
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
        emb_path = config.file_path_summary
        if os.path.exists(emb_path):
            try:
                with open(emb_path, "r", encoding="utf-8") as f:
                    self.summary_dict= json.load(f)
                print(f"成功加载历史embedding记录，共 {len(self.summary_dict)} 份文档")
            except Exception as e:
                print(f"embedding.json读取失败，重置为空: {e}")
                self.summary_dict= {}
        else:
            self.summary_dict= {}

    def _load_embedding_store(self):
        """从json加载历史文档向量，仅在__init__执行"""
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
        chunks=[c for c in chunks if len(c.strip())>=drop_below]
        merged=[]
        buf=""
        for chunk in chunks:
            buf+=chunk
            if len(buf)>=min_size:
                merged.append(buf)
                buf=""
        if buf:
            if not merged:
                merged.append(buf)      # 整页都很短：不能丢，单独成块
            elif len(merged[-1])+len(buf)<=max_size:
                merged[-1]+=buf
            else:
                merged.append(buf)
        return merged
    def get_final_chunks(self,text):
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
        # 空文本保护
        if not page_texts or all(len(t.strip()) == 0 for t in page_texts):
            return True, 0.0,None

        # 当前文档均值向量，用于相似度比对
        doc_embed = np.mean(page_embedding, axis=0).reshape(1, -1)
        max_sim = 0.0
        file_name=None

        # 遍历历史文档，逐个计算均值向量相似度
        for old_file_name,old_file_embeds in self.embeddings_dict.items():
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
            info = extract_from_pages(data, filename)
            try:
                from model_extractor import enrich
                info = enrich(info)
            except Exception as e:
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
        """私有方法：只做pdf解析，返回data，只写这一处解析逻辑"""
        a = base64.b64decode(base64_str.encode("utf-8"))
        file_pdf_doc = pymupdf.open("pdf", a)
        data: list[dict] = pymupdf4llm.to_markdown(file_pdf_doc, page_chunks=True)
        file_pdf_doc.close()
        return data

    def _save_to_all_db(self,filename:str,data,page_embedding,str_hashlib):
        """公共入库：Chroma、摘要、md5、embeddings_dict、bm25统一写入"""
        # 追加当前文档原始每页embedding
        self.embeddings_dict[filename] = page_embedding
        # 持久化写入json
        self._save_embedding_store()
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
        # 保存入库文档的md5值
        save_md5(str_hashlib,filename)
        # 将分词加入bm25模型中
        self.build_bm25(all_chunks,filename,page_nums)
        return True, None, filename  # (入库是否成功，旧的文件名，新的文件名)
    def upload_str_pdf(self,base64_str:str,filename:str):
        str_hashlib=get_str_md5(base64_str)
        file_exist()

        if check_md5(str_hashlib):
            return False,filename,filename#(入库是否成功，旧的文件名，新的文件名)

        data = self._parse_pdf_base64(base64_str)
        #拼接多页文档内容，对所有文档进行向量转化
        page_texts=[]
        for i in data:
            pages=i["text"]
            page_texts.append(pages)
        page_embedding=self.embeddings.embed_documents(page_texts)
        #解析
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
            from agent import agent
            try:
                with open(config.file_path_summary,"r",encoding="utf-8") as f:
                    all_doc=json.load(f)
                    old_file_summary=all_doc[old_file_name]
            except (FileNotFoundError, KeyError, json.JSONDecodeError) as e:
                raise Exception(f"读取旧文档摘要失败: {e}")
            new_file_summary=self.simple_extract_summary(data[0]["text"])
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

    def real_save_pdf(self,filename:str):#(在高相似度的情况下任然决定文档入库)
        cache=self.temp_pdf[filename]
        data = cache["data"]
        filename = cache["filename"]
        md5_str = cache["md5"]
        file_embedding=cache["file_embedding"]
        result, old_file_name, new_file_name = self._save_to_all_db(filename, data, file_embedding, md5_str)
        del self.temp_pdf[filename]
        return result, old_file_name, new_file_name#(入库是否成功，旧的文件名，新的文件名)

    def delete_file(self,filename:str):
        """删除embedding文件，summary文件，向量数据库，bm25模型,md5文件中的所属内容"""
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
        old_chroma_data=self.chroma.get(where={"source":filename})
        self.chroma.delete(where={"source":filename})
        #4.bm25模型
        new_corpus = []
        new_source = []
        for token, src in zip(self.bm25_corpus, self.bm25_chunk_source):
            if src != filename:
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
    import pymupdf4llm
    service=Knolege()
    file=r"C:\Users\34078\Desktop\1706.03762v1.pdf"
    filename="1706.03762v1.pdf"
    with open(file,"rb") as f:
        data=f.read()
    base64_str = base64.b64encode(data).decode("utf-8")
    a=service.upload_str_pdf(base64_str,filename)
    print(a)