#!/usr/bin/env python3
# ssd3f_patch.py (2026-11-19, mikro/q4h32/; after ssd_patch.py and ssd3e_patch.py; the user: "64gb ram q4 polne verzije ce je opcija
# da se tudi kontekst poveca na 512k"): the K/V pool's lent expert slots and the RAM tier. Without it a long context's K/V takes the
# arena pool's slots and their experts are left on the disk alone (no host copies with FNX_SSD_EXPERTS), so writing deep in a 512k
# context reads the hottest experts from the disk; and slots given back are filled from the disk.
#   - Engine::arena_set, lending: the slots' experts copied down into the RAM tier first (p100_dma_ssd_save, as a read's prompt slots)
#   - Engine::arena_refill: an expert the RAM tier holds is copied up from there, then let go there (p100_dma_ssd_drop); the rest are
#     read from the file as before
# Off (FNX_SSD_EXPERTS or FNX_SSD_RAM_GIB unset, or no copier): every changed line does what it did.
import sys
from pathlib import Path

S = Path("src")
if "ssd3_patch.py" not in (S / "p100_dma.cuh").read_text(encoding="utf-8") or "SSD_SPARE_NEED" not in (S / "p100_dma.cuh").read_text(encoding="utf-8"):
    sys.exit("ssd3f_patch.py: ssd_patch.py and ssd3e_patch.py first")
if "ssd3f_patch.py" in (S / "engine.cu").read_text(encoding="utf-8"):
    sys.exit("ssd3f_patch.py: applied already")


def rep(name, old, new, count=1):
    p = S / name
    t = p.read_text(encoding="utf-8")
    n = t.count(old)
    if n != count:
        sys.exit(f"ssd3f_patch.py: {name}: {n} matches, want {count}:\n{old[:300]}")
    p.write_text(t.replace(old, new), encoding="utf-8")


# ---------------------------------------------------------------- p100_dma_ssd_drop
rep("p100_dma.cuh",
    "void p100_dma_wait(void* h, void* st) {\n",
    "// ssd3f_patch.py: experts copied up from the RAM tier by the engine (their copies landed) let go there; the copier idle\n"
    "void p100_dma_ssd_drop(void* h, const std::vector<int>& gs) {\n"
    "    auto* c = (DmaCard*) h;\n"
    "    SsdCache* sc = c && ssd_ram_gib() > 0 ? dma_ssd_cache(c) : nullptr;\n"
    "    if (!sc) return;\n"
    "    std::lock_guard<std::mutex> lk(sc->mu);\n"
    "    for (const int g : gs) sc_drop(sc, g);\n"
    "}\n"
    "\n"
    "void p100_dma_wait(void* h, void* st) {\n")
rep("p100_over.hpp",
    "const char* p100_dma_ssd_find(void* h, int g);\n",
    "const char* p100_dma_ssd_find(void* h, int g);\n"
    "// ssd3f_patch.py: experts the engine copied up from the RAM tier (landed) let go there\n"
    "void p100_dma_ssd_drop(void* h, const std::vector<int>& gs);\n")

# ---------------------------------------------------------------- arena_set: the lent slots' experts down into the RAM tier
rep("engine.cu",
    "        arena_slots(ec, from, to, b.st);\n"
    "        ck(cudaStreamSynchronize(b.st), \"arena: slots\");\n",
    "        if (to > from && ssd_on() && ssd_ram_gib() > 0 && b.dma) {   // ssd3f_patch.py: the lent slots' experts down into the RAM tier first\n"
    "            const int p = ec.arena_pool;\n"
    "            std::vector<int> own((size_t) (to - from));\n"
    "            ck(cudaMemcpy(own.data(), ec.owner + ec.s0[p] + from, own.size() * sizeof(int), cudaMemcpyDeviceToHost), \"ssd: the lent slots' owners\");\n"
    "            std::vector<SsdFill> keep;\n"
    "            for (int i = from; i < to; ++i)\n"
    "                if (const int g = own[(size_t) (i - from)]; g >= 0) {\n"
    "                    char* at = ec.pool[p] + (size_t) i * ec.slot_bytes[p];\n"
    "                    keep.push_back({g, at, at + ec.dn_off[p], ec.dn_off[p], ec.slot_bytes[p] - ec.dn_off[p]});\n"
    "                }\n"
    "            p100_dma_ssd_save(b.dma, keep, b.st);\n"
    "        }\n"
    "        arena_slots(ec, from, to, b.st);\n"
    "        ck(cudaStreamSynchronize(b.st), \"arena: slots\");\n")

# ---------------------------------------------------------------- arena_refill: from the RAM tier first
rep("engine.cu",
    "    std::vector<SsdFill> refill;   // ssd_patch.py: the experts with no host copy, read from the file after the loop\n",
    "    std::vector<SsdFill> refill;   // ssd_patch.py: the experts with no host copy, read from the file after the loop\n"
    "    std::vector<int> from_tier;    // ssd3f_patch.py: those copied up from the RAM tier, let go there after the loop\n")
rep("engine.cu",
    "        if (ssd_tagged(w.ec_host_gu[(size_t) i])) {\n"
    "            refill.push_back({i, at, at + ec.dn_off[p], L.gu_e_stride, L.down_e_stride});\n"
    "        } else {\n",
    "        if (ssd_tagged(w.ec_host_gu[(size_t) i])) {\n"
    "            if (const char* hc = ssd_ram_gib() > 0 && cb_[c]->dma ? p100_dma_ssd_find(cb_[c]->dma, i) : nullptr) {   // ssd3f_patch.py\n"
    "                ck(cudaMemcpy(at, hc, L.gu_e_stride, cudaMemcpyDefault), \"arena: refill\");\n"
    "                ck(cudaMemcpy(at + ec.dn_off[p], hc + L.gu_e_stride, L.down_e_stride, cudaMemcpyDefault), \"arena: refill\");\n"
    "                from_tier.push_back(i);\n"
    "            } else\n"
    "            refill.push_back({i, at, at + ec.dn_off[p], L.gu_e_stride, L.down_e_stride});\n"
    "        } else {\n")
rep("engine.cu",
    "    if (!refill.empty()) {   // ssd_patch.py\n",
    "    if (!from_tier.empty()) {   // ssd3f_patch.py: they live on the card now\n"
    "        p100_dma_ssd_drop(cb_[c]->dma, from_tier);\n"
    "        std::fprintf(stderr, \"fnx ssd: card %d: %zu of the arena's slots filled again from the RAM tier, %zu from the file\\n\", c, from_tier.size(),\n"
    "                     refill.size());\n"
    "    }\n"
    "    if (!refill.empty()) {   // ssd_patch.py\n")
print("ssd3f_patch.py: the K/V pool's lent slots' experts into the RAM tier, the arena's refill from it (p100_dma.cuh, p100_over.hpp, engine.cu)")
