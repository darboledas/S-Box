#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate PRNG-initialized 8x8 S-box JSON inputs compatible with EVO-SBOX_v5.py.

python gen_prng_inputs_for_evo_sbox_v5.py --outdir prng_inputs --count 10 --base-seed 123456


EVO-SBOX_v5 loads S0 from:
  data.get("sbox", data.get("sbox_original", data.get("sbox_evo_v40_academic", ...)))

So we MUST provide at least:
  { "sbox": [ ... 256 ints ... ] }

We also include optional metadata for traceability:
- timestamp_utc
- source (PRNG info)
- seed_u64
- sha256 (hash of bytes(sbox))
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import random
import secrets
from typing import List, Dict, Any


def utc_now_z() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_sbox(sbox: List[int]) -> str:
    return hashlib.sha256(bytes(sbox)).hexdigest()


def make_perm_mt19937(seed_u64: int) -> List[int]:
    rng = random.Random(seed_u64)  # MT19937
    s = list(range(256))
    rng.shuffle(s)
    return s


def derive_seed_list(n: int, base_seed: int | None) -> List[int]:
    if base_seed is None:
        return [secrets.randbits(64) for _ in range(n)]
    # deterministic derivation from base_seed
    base = base_seed.to_bytes(16, "big", signed=False)
    out: List[int] = []
    for i in range(n):
        h = hashlib.sha256(base + i.to_bytes(4, "big")).digest()
        out.append(int.from_bytes(h[:8], "big", signed=False))
    return out


def build_json(seed_u64: int, sbox: List[int], idx: int) -> Dict[str, Any]:
    return {
        # EVO-SBOX_v5 will read this:
        "sbox": sbox,

        # optional but useful:
        "timestamp_utc": utc_now_z(),
        "sha256": sha256_sbox(sbox),
        "artifact_id": f"PRNG_MT19937_{idx:02d}",
        "source": {
            "type": "PRNG",
            "engine": "python.random.Random",
            "algorithm": "MT19937",
            "seed_u64": seed_u64,
        },
        "passed": False  # EVO-SBOX_v5 sets this to True after optimization; optional
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate PRNG S-box JSON inputs for EVO-SBOX_v5.")
    ap.add_argument("--outdir", default="prng_inputs", help="Output directory for JSON files.")
    ap.add_argument("--count", type=int, default=10, help="How many S-box JSON files to generate.")
    ap.add_argument("--base-seed", type=int, default=None,
                    help="Optional integer base seed to reproduce the full set deterministically.")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    seeds = derive_seed_list(args.count, args.base_seed)

    for idx, seed in enumerate(seeds, start=1):
        sbox = make_perm_mt19937(seed)
        data = build_json(seed, sbox, idx)

        path = os.path.join(args.outdir, f"prng_sbox_{idx:02d}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    print(f"OK: wrote {args.count} JSON files to {os.path.abspath(args.outdir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
