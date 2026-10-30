import os
import hashlib
import config

# md5.txt 是"哪些文件已经入过库"的唯一账本，格式是一行一条：
#     <md5 指纹>,<文件名>
# 为什么两列都要：指纹用来判断"内容是否重复"，文件名用来在删除时反查。
# 只存指纹的话，删文件时没法知道要删哪一行。


def file_exist():
    """确认 md5.txt 存在。首次运行时顺手建一个空文件，
    这样后面所有地方都能直接用追加/读取模式，不用每处都判存在。
    """
    if os.path.exists(config.md5_path):
        return True
    else:
        with open(config.md5_path,"w",encoding="utf-8"):
            pass
        return False


def get_str_md5(input_str:str):
    """算 base64 串的 md5，作为文件指纹。

    用 base64 而不是原始二进制来算，是因为上传走的是 base64 文本，
    两边算出来的结果才能对上；同一份 PDF 重新编码一次 base64 结果是不变的。
    """
    hash_md5=hashlib.md5()
    str_bytes=input_str.encode("utf-8")
    hash_md5.update(str_bytes)
    return hash_md5.hexdigest()


def check_md5(md5_str:str):
    """逐行比对指纹，命中说明这份文件已经入过库，直接跳过上传。

    这里没有把 md5.txt 全读成字典，而是边读边比、命中就返回：
    文件通常只有几十行，但这样写不会因为文件变大而多吃内存。
    """
    if not os.path.exists(config.md5_path):
        return False
    with open(config.md5_path,"r",encoding="utf-8") as f:
        for line in f.readlines():
            line=line.strip()
            if not line:
                continue
            md5, fname = line.split(",",1) #只分割一次，文件名含逗号也不怕
            if md5_str == md5:
                return True
    return False


def save_md5(md5_str:str, filename:str):
    """追加一条登记。调用点必须放在入库流程的最后一步——
    提前写的话，中途失败会留下"账本里有、数据里没有"的幽灵记录，
    下次重传会被自己挡住，还读不到对应的摘要。
    """
    with open(config.md5_path,"a",encoding="utf-8") as f:
        f.write(f"{md5_str},{filename}\n")


def remove_md5_by_filename(target_filename:str):
    """删除某个文件的所有登记行（重新保存时会调用，先删旧的再写新的）。

    做法是重建：把不要删的行收集起来，最后整体覆盖写回。
    直接改文件中间几行在文本文件里做不到，只能这样。
    """
    if not os.path.exists(config.md5_path):
        return
    new_lines = []
    with open(config.md5_path,"r",encoding="utf-8") as f:
        for line in f.readlines():
            line = line.strip()
            if not line:
                continue
            md5, fname = line.split(",",1)
            if fname != target_filename:
                # 不是要删的文件，保留
                new_lines.append(line)
    # 重写文件，覆盖
    with open(config.md5_path,"w",encoding="utf-8") as f:
        for l in new_lines:
            f.write(l + "\n")
