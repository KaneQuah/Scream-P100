#!/usr/bin/env python3
"""UNVERIFIED PROOF OF CONCEPT, nobody has run it on four cards: writes a run config for the 4-card engine (bash build.sh --4x).

    python3 engine-4x/fill_profile_4x.py --models /data/flash-next/models --mtp /data/flash-next/prep/mtp-rt \
        --tokenizer /data/flash-next/prep/tokenizer --logs /data/flash-next/logs --out /data/flash-next/profile-4x.json
    bash run.sh /data/flash-next --config /data/flash-next/profile-4x.json

It fills one of the IQ3_S templates beside this script through profile/fill_profile.py (which fills in and checks every path,
and writes nothing when one is missing), then names the four cards. ISTA-DASLab's IQ3_S only: UD-Q4_K_XL's experts (72 GiB) would
leave too many of them in RAM behind a shared PCIe 3.0 x4 link, and Edge0-145B runs on two cards only (docs/POC-4X.md).

--context    256k (default): 262,144 positions, the model's own, no YaRN; the K/V pool, which grows into the experts' VRAM slots
             as a conversation gets longer (FNX_ARENA), takes at most half of what 512k takes from them, about 1.5 GiB a card.
             512k: 524,288 positions with YaRN x2 (our production's), up to about 3 GiB a card. 128k: 131,072, no YaRN
--gpu        the four P100 as nvidia-smi numbers them (default "0,1,2,3")
--exe        the engine (default build-4x/split/strata-split-4)
--models, --mtp, --tokenizer, --vision-exe, --logs, --route-prior, --draft-vocab: as profile/fill_profile.py
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RELEASE = HERE.parent


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--models", required=True)
    ap.add_argument("--mtp", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--vision-exe")
    ap.add_argument("--logs", default="/var/tmp")
    ap.add_argument("--route-prior", default="0")
    ap.add_argument("--draft-vocab")
    ap.add_argument("--exe", default=str(RELEASE / "build-4x" / "split" / "strata-split-4"))
    ap.add_argument("--gpu", default="0,1,2,3")
    ap.add_argument("--context", choices=["128k", "256k", "512k"], default="256k")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    cards = [c.strip() for c in a.gpu.split(",")]
    if len(cards) != 4 or not all(c.isdigit() for c in cards) or len(set(cards)) != 4:
        sys.exit("--gpu: four different card numbers, e.g. 0,1,2,3")
    template = HERE / f"flash-next-iq3s-4x-{'512k' if a.context == '512k' else '256k'}.template.json"
    out = Path(a.out)
    tmp = out.with_name(out.name + ".fill-tmp")
    cmd = [sys.executable, str(RELEASE / "profile" / "fill_profile.py"), "--models", a.models, "--mtp", a.mtp, "--tokenizer", a.tokenizer,
           "--logs", a.logs, "--route-prior", a.route_prior, "--exe", a.exe, "--gpu", "0,1", "--template", str(template), "--out", str(tmp)]
    if a.vision_exe:
        cmd += ["--vision-exe", a.vision_exe]
    if a.draft_vocab:
        cmd += ["--draft-vocab", a.draft_vocab]
    r = subprocess.run(cmd, stdout=subprocess.DEVNULL)   # its own "wrote" line would name two cards; its errors still show
    if r.returncode != 0:
        sys.exit(r.returncode)
    cfg = json.loads(tmp.read_text(encoding="utf-8"))
    os.remove(tmp)
    cfg["gpu"] = ",".join(cards)
    if a.context == "128k":
        cfg["args"][cfg["args"].index("--max-context") + 1] = "131072"
    out.write_text(json.dumps(cfg, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {out}: UNVERIFIED 4-card proof of concept, IQ3_S, cards {cfg['gpu']}, {a.context}, "
          f"FNX_ROUTE_PRIOR={cfg['env']['FNX_ROUTE_PRIOR']}, pictures {'on' if 'vision' in cfg else 'off'}, engine {cfg['exe']}")


if __name__ == "__main__":
    main()
