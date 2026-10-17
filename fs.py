def _merge_small_chunks(chunks, min_size=800, max_size=1500, drop_below=20):
    """语义块 -> 合并到 min_size；并把页码之类的碎块处理掉。"""
    chunks = [c for c in chunks if len(c.strip()) >= drop_below]  # 页码页眉页脚
    merged, buf = [], ""
    for ch in chunks:
        buf += ch
        if len(buf) >= min_size:
            merged.append(buf)
            buf = ""
    if buf:                                    # 尾巴并回上一个，超 max_size 才单独成块
        if merged and len(merged[-1]) + len(buf) <= max_size:
            merged[-1] += buf
        else:
            merged.append(buf)
    return merged