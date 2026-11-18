"""Deterministic sparse top-k oracle for synthetic MTI validation only.

Native BF16 top-k can choose different keys tied at its threshold. A stable
logical-position tie rule lets the adapter be compared to a native scheduler
continuation without that noise. These patches are removed before benchmarking.
"""


def install_stable_topk():
    import torch
    from sglang.srt.layers.attention.dsv4.v41_indexer import full_topk, sparse_table

    patches = []

    def replace(module, name, function):
        patches.append((module, name, getattr(module, name)))
        setattr(module, name, function)

    def stable_order(logits, lens, width):
        n = int(lens.max())
        scores = logits[:, :n].float()
        valid = torch.arange(n, device=scores.device)[None, :] < lens[:, None]
        scores = scores.masked_fill(~valid, -torch.inf)
        order = scores.argsort(dim=-1, descending=True, stable=True)[:, :min(n, width)]
        return order, order < lens[:, None]

    def paged(logits, lens, table, out, page_size, *unused, raw=None):
        order, valid = stable_order(logits, lens, out.shape[-1])
        pages = table.gather(1, (order // page_size).long())
        slots = pages.long() * page_size + order % page_size
        out.fill_(-1)
        out[:, :order.shape[-1]].copy_(torch.where(valid, slots, -1).to(out.dtype))
        if raw is not None:
            raw.fill_(-1)
            raw[:, :order.shape[-1]].copy_(torch.where(valid, order, -1).to(raw.dtype))

    def from_metadata(logits, metadata, page_indices, raw_indices=None, *,
                      rows=None, topk_metadata=None):
        rows = slice(None) if rows is None else rows
        paged(logits, metadata.compressed_seq_lens[rows], metadata.page_table[rows],
              page_indices[rows], metadata.compressed_page_size,
              raw=raw_indices[rows] if raw_indices is not None else None)

    def sparse(logits, lens, blocks, out):
        order, valid = stable_order(logits, lens, out.shape[-1])
        chosen = blocks.gather(1, (order // 8).long()).long()
        slots = chosen * 8 + order % 8
        out.fill_(-1)
        out[:, :order.shape[-1]].copy_(torch.where(valid, slots, -1).to(out.dtype))

    replace(full_topk, 'topk_transform_paged_from_metadata', from_metadata)
    replace(sparse_table, 'topk_transform_paged_bf16', paged)
    replace(sparse_table, 'topk_transform_paged_v2', paged)
    replace(sparse_table, 'topk_transform_sparse', sparse)

    def restore():
        for module, name, function in reversed(patches):
            setattr(module, name, function)

    return restore
