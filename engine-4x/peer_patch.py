#!/usr/bin/env python3
# UNVERIFIED PROOF OF CONCEPT: nobody has run this on four cards (engine-4x/README.md).
#
# peer_patch.py (2026-11-19, the 4-card proof of concept): FNX_PEER on NVIDIA. The window sums and the attention merge of the
# decode windows are pushed into the other cards' device memory when every pair of cards passes a start probe (cuda_peer.hpp),
# through the push kernels the engine already has for AMD (k_ring_a2a_push, k_att_merge_a2a with device mailboxes); otherwise,
# or with FNX_PEER=0, the host exchange as in every other build, and one line in the log says which and why. The prompt path is
# not changed: its sums and its attention exchange still go through page-locked host memory (docs/POC-4X.md says why).
# Run in the engine's source folder (the one holding src/); cuda_peer.hpp beside this script goes to src/.
import shutil
import sys
from pathlib import Path

S = Path("src")
if not (S / "engine.cu").exists():
    sys.exit("peer_patch.py: run it in the folder that holds src/")


def rep(name, old, new, count=1):
    p = S / name
    t = p.read_text(encoding="utf-8")
    n = t.count(old)
    if n != count:
        sys.exit(f"peer_patch.py: {name}: {n} matches, want {count}:\n{old[:300]}")
    p.write_text(t.replace(old, new), encoding="utf-8")


if "peer_patch.py" in (S / "engine.cu").read_text(encoding="utf-8"):
    sys.exit("peer_patch.py: applied already")
shutil.copy(Path(__file__).resolve().parent / "cuda_peer.hpp", S / "cuda_peer.hpp")

rep("engine.cu",
    '#if FNX_HIP\n#include "hip/peer.h"\n#include "hip/rccl.h"\n#endif\n',
    '#if FNX_HIP\n#include "hip/peer.h"\n#include "hip/rccl.h"\n#else\n#include "cuda_peer.hpp"   // peer_patch.py (engine-4x)\n#endif\n')

# the decision and the mailbox allocator on NVIDIA, after the AMD block
rep("engine.cu",
    "        own_dev(p);\n"
    "        return p;\n"
    "    };\n"
    "#endif\n"
    "    load_cvec();\n",
    "        own_dev(p);\n"
    "        return p;\n"
    "    };\n"
    "#else\n"
    "    // peer_patch.py (engine-4x, NOT RUN on four cards): the same choice on NVIDIA, by hip/peer_policy.hpp's rules. FNX_PEER=0: the\n"
    "    // host exchange, no probe. Unset or 1: the cards' IOMMU groups read, peer access enabled between every pair and every directed\n"
    "    // pair probed (cuda_peer.hpp); any failure leaves the host exchange for the whole run, with one line saying why. FNX_PEER_PARTS\n"
    "    // as on AMD: 1 the window sums, 2 the attention merge, 3 both (the default). The prompt path keeps the host exchange either way\n"
    "    {\n"
    "        std::vector<int> devs;\n"
    "        for (int c = 0; c < NCARD; ++c) devs.push_back(m.cards[c].dev);\n"
    "        const PeerDecision pd = peer_decide(std::getenv(\"FNX_PEER\"), PEER_DEFAULT_ON,\n"
    "                                            [&](std::string& why) { return cuda_peer_enable_and_probe(devs, why); });\n"
    "        if (!pd.line.empty()) std::fprintf(stderr, \"fnx: %s\\n\", pd.line.c_str());\n"
    "        else std::fprintf(stderr, \"fnx: FNX_PEER=0: the host exchange, no peer probe\\n\");\n"
    "        peer_on = pd.use;\n"
    "        peer_parts = peer_on ? (int) env_int(\"FNX_PEER_PARTS\", 3) : 0;\n"
    "        if (peer_on && (peer_parts < 1 || peer_parts > 3)) throw std::runtime_error(\"FNX_PEER_PARTS: 1, 2 or 3\");\n"
    "        if (peer_on)\n"
    "            std::fprintf(stderr, \"fnx: peer push: %s%s%s pushed into the peers' memory\\n\", peer_parts & 1 ? \"the window sums\" : \"\",\n"
    "                         peer_parts == 3 ? \" and \" : \"\", peer_parts & 2 ? \"the attention merge\" : \"\");\n"
    "    }\n"
    "    // a mailbox the other cards store into: device memory on the current card, zeroed (peer access is on between every pair)\n"
    "    auto pxalloc = [&](size_t bytes) -> void* { return (void*) dalloc<unsigned char>(bytes); };\n"
    "#endif\n"
    "    load_cvec();\n")

# the mailboxes: the AMD branch's allocation on NVIDIA too
rep("engine.cu",
    "#if FNX_HIP\n"
    "        if (peer_on && (peer_parts & 1)) b->ring.pk = pxalloc((size_t) 2 * NCARD * MAXT * (N / 2) * sizeof(unsigned long long));\n"
    "        b->att_ring.in = peer_on && (peer_parts & 2) ? (unsigned long long*) pxalloc(att_ring_words() * sizeof(unsigned long long))\n"
    "                                                     : hxalloc<unsigned long long>(att_ring_words());\n"
    "#else\n"
    "        b->att_ring.in = hxalloc<unsigned long long>(att_ring_words());\n"
    "#endif\n",
    "        // peer_patch.py: the AMD branch's mailboxes on NVIDIA too (device memory when the peer push is on, else host memory)\n"
    "        if (peer_on && (peer_parts & 1)) b->ring.pk = pxalloc((size_t) 2 * NCARD * MAXT * (N / 2) * sizeof(unsigned long long));\n"
    "        b->att_ring.in = peer_on && (peer_parts & 2) ? (unsigned long long*) pxalloc(att_ring_words() * sizeof(unsigned long long))\n"
    "                                                     : hxalloc<unsigned long long>(att_ring_words());\n")

rep("engine.cu",
    "    std::fprintf(stderr, \"fnx: HOSTRING build: the cards exchange through page-locked host memory; peer access is never enabled, \"\n"
    "                         \"no peer probe, IOMMU and BAR1 checks skipped\\n\");\n",
    "    if (peer_on)   // peer_patch.py\n"
    "        std::fprintf(stderr, \"fnx: engine-4x: the decode windows' %s go card to card; the prompt path exchanges through page-locked host \"\n"
    "                             \"memory\\n\", peer_parts == 3 ? \"sums and attention merge\" : peer_parts & 1 ? \"sums\" : \"attention merge\");\n"
    "    else\n"
    "        std::fprintf(stderr, \"fnx: engine-4x: the cards exchange through page-locked host memory (no peer access)\\n\");\n")
print("peer_patch.py: FNX_PEER on NVIDIA (cuda_peer.hpp)")
