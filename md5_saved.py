import os
import hashlib
import config
def file_exist():
    if os.path.exists(config.md5_path):
        return True
    else:
        with open(config.md5_path,"w",encoding="utf-8"):
            pass
        return False
def get_str_md5(input_str:str):
    hash_md5=hashlib.md5()
    str_bytes=input_str.encode("utf-8")
    hash_md5.update(str_bytes)
    return hash_md5.hexdigest()
def check_md5(md5_str:str):
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
    with open(config.md5_path,"a",encoding="utf-8") as f:
        f.write(f"{md5_str},{filename}\n")
def remove_md5_by_filename(target_filename:str):
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