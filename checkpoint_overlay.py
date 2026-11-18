"""Load pqhaz's 1017 tensor replacements without copying or editing base shards.

The official checkpoint is always opened by SGLang's native loader. Only the
38 named CPU tensors are replaced before native TP sharding and FP8 processing.
The DSpark draft model is deliberately outside this hook.
"""
import hashlib
import json
import os
from pathlib import Path
import struct
import time

ROOT = Path(__file__).resolve().parent
REPO = 'pqhaz/DeepSeek-V4.1-Flash-1017-heretic'
REVISION = '12d58ad0878b5008e80d720251adb969a0069412'
BASE_REVISION = '1bae0e1088993b9b82de08f59de5be5a34e42170'
PATCH_SHA256 = '16e63cd3e79c3590855aad340b5b9dc37f27562bf2d0a40f85268915da221ce2'
PATCH = ROOT / '.runs/heretic/source' / REVISION / 'patch/heretic_patch.safetensors'
EXPECTED_KEYS = {f'layers.{i}.attn.wo_b.weight' for i in range(2, 40)}


def safetensors_header(path):
    with Path(path).open('rb') as handle:
        length = struct.unpack('<Q', handle.read(8))[0]
        if not 0 < length <= 64 * 1024 * 1024:
            raise ValueError(f'Invalid safetensors header length: {path}')
        return json.loads(handle.read(length))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def validate_patch(base, patch=PATCH):
    base, patch = Path(base), Path(patch)
    verified = json.loads((base / '.checkpoint-verified.json').read_text())
    if verified.get('revision') != BASE_REVISION:
        raise ValueError('This patch requires the verified official 1017 revision')
    if sha256(patch) != PATCH_SHA256:
        raise ValueError('Heretic patch SHA256 does not match the pinned release')
    header = safetensors_header(patch)
    keys = set(header) - {'__metadata__'}
    if keys != EXPECTED_KEYS:
        raise ValueError('Expected exactly the 38 attention output tensors in blocks 2–39')
    index = json.loads((base / 'model.safetensors.index.json').read_text())['weight_map']
    headers = {}
    for name in sorted(keys):
        shard = index[name]
        if shard not in headers:
            headers[shard] = safetensors_header(base / shard)
        original, replacement = headers[shard][name], header[name]
        if (original['shape'] != replacement['shape'] or
                original['dtype'] != replacement['dtype'] or
                original['dtype'] != 'F8_E4M3'):
            raise ValueError(f'Patch shape/dtype mismatch: {name}')
    return {'repo': REPO, 'revision': REVISION, 'sha256': PATCH_SHA256,
            'patch': str(patch.resolve()), 'base': str(base.resolve()),
            'tensor_names': sorted(keys), 'tensor_count': len(keys),
            'patch_bytes': patch.stat().st_size}


def replacement_weights(weights, patch, seen):
    """Preserve names/order and replace only matching native checkpoint tensors."""
    from safetensors import safe_open
    with safe_open(str(patch), framework='pt', device='cpu') as replacements:
        for name, weight in weights:
            if name in EXPECTED_KEYS:
                if name in seen:
                    raise RuntimeError(f'Duplicate checkpoint tensor: {name}')
                replacement = replacements.get_tensor(name)
                if replacement.shape != weight.shape or replacement.dtype != weight.dtype:
                    raise RuntimeError(f'Runtime tensor mismatch: {name}')
                seen.add(name)
                yield name, replacement
            else:
                yield name, weight


def install(patch, base, audit_dir):
    from functools import wraps
    from sglang.srt.model_loader.loader import DefaultModelLoader
    original = DefaultModelLoader.load_weights_only
    if getattr(original, '_ds41_checkpoint_overlay', False):
        return

    @wraps(original)
    def load(model, weights, target_device):
        # Both the language model and DSpark pass through this loader. Apply
        # only to the target; never modify the draft, vision, or other models.
        if type(model).__name__ != 'DeepseekV4ForCausalLM':
            return original(model, weights, target_device)
        if getattr(model.config, 'model_type', None) != 'deepseek_v41':
            raise RuntimeError('The checkpoint overlay supports only DeepSeek V4.1')
        metadata = validate_patch(base, patch)
        seen = set()
        result = original(model, replacement_weights(weights, patch, seen), target_device)
        if seen != EXPECTED_KEYS:
            raise RuntimeError(f'Incomplete Heretic overlay: loaded {len(seen)}/38 tensors')
        import torch.distributed as distributed
        rank = distributed.get_rank() if distributed.is_initialized() else 0
        destination = Path(audit_dir)
        destination.mkdir(parents=True, exist_ok=True)
        record = dict(metadata, pid=os.getpid(), rank=rank, loaded_at=time.time(),
                      model_class=type(model).__name__, applied_tensor_names=sorted(seen))
        temporary = destination / f'rank-{rank}-{os.getpid()}.tmp'
        temporary.write_text(json.dumps(record, indent=2) + '\n')
        temporary.replace(temporary.with_suffix('.json'))
        print(f'Heretic checkpoint overlay: rank {rank}, {len(seen)}/38 tensors loaded', flush=True)
        return result

    load._ds41_checkpoint_overlay = True
    DefaultModelLoader.load_weights_only = staticmethod(load)
