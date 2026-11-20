#!/usr/bin/env python3
# ff64_patch.py: the experts' FF width on four cards in 64-column units (192, 192, 128, 128) instead of 32-column ones (160 each).
# Why: the P100 down kernels want slices in units of 64 columns (p100_moe.cu: "columns not in units of 64"), and with 160-column
# slices (tail_patch.py zero-pads the 32-column tail of the prompt path's down kernel) the first-token logits of the 4-card build
# were wrong on a real run (top token '<think>' / 'The' where llama.cpp and the 2-card engine say 'We'), on the host exchange and
# on peer push, ring and direct gather alike; with 64-column slices they match both (docs: issue #1). The cost is an uneven split:
# cards 0 and 1 carry 50% more FF columns than cards 2 and 3. Two and three cards are unchanged. Run first, before the other
# patches (every slice is then a multiple of 64, so tail_patch.py's tail path is not used), in the engine's source folder.
import sys
from pathlib import Path

S = Path("src")
if not (S / "card_split.hpp").exists():
    sys.exit("ff64_patch.py: run it in the folder that holds src/")


def rep(name, old, new):
    p = S / name
    t = p.read_text(encoding="utf-8")
    n = t.count(old)
    if n != 1:
        sys.exit(f"ff64_patch.py: {name}: {n} matches, want 1:\n{old[:300]}")
    p.write_text(t.replace(old, new), encoding="utf-8")


if "ff_unit" in (S / "card_split.hpp").read_text(encoding="utf-8"):
    sys.exit("ff64_patch.py: applied already")

rep("card_split.hpp",
    "constexpr int ff_lo(int n, int c) { return 32 * (c * (FF / 32 / n) + (c < FF / 32 % n ? c : FF / 32 % n)); }\n",
    "// ff64_patch.py: four cards cut FF in 64-value blocks (3, 3, 2, 2 of them: 192, 192, 128, 128), every other count in 32-value blocks as before\n"
    "constexpr int ff_unit(int n) { return n == 4 ? 64 : 32; }\n"
    "constexpr int ff_lo(int n, int c) { return ff_unit(n) * (c * (FF / ff_unit(n) / n) + (c < FF / ff_unit(n) % n ? c : FF / ff_unit(n) % n)); }\n")
rep("card_split.hpp",
    "if (ff_lo(n, c) % 32 || ff_hi(n, c) <= ff_lo(n, c)) return false;",
    "if (ff_lo(n, c) % ff_unit(n) || ff_hi(n, c) <= ff_lo(n, c)) return false;")
rep("model.hpp",
    "constexpr int FF_CARD_MAX = (FF + 32 * NCARD - 1) / (32 * NCARD) * 32;\n",
    "constexpr int FF_UNIT_ = NCARD == 4 ? 64 : 32;   // ff64_patch.py: card_split.hpp ff_unit\n"
    "constexpr int FF_CARD_MAX = (FF + FF_UNIT_ * NCARD - 1) / (FF_UNIT_ * NCARD) * FF_UNIT_;\n")
