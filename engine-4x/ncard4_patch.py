#!/usr/bin/env python3
# UNVERIFIED PROOF OF CONCEPT: nobody has run this on four cards (engine-4x/README.md).
#
# ncard4_patch.py (2026-11-19, the 4-card proof of concept): the three places where engine/ is written for two cards only.
#   - pf_gather (FNX_PF_SP): each card's finished part of a chunk now goes once to its own host rows and every other card copies it
#     from there (NCARD - 1 senders, a flag each), instead of into the one other card's rows. On two cards it moves the same bytes
#     in the same order as before. FNX_PF_SP is off in the 4-card templates (it needs FNX_PF_GATHER=ring and moves more bytes over
#     the host link than it saves; docs/POC-4X.md), so this keeps the switch working rather than making it the default
#   - pf_lat_swap (FNX_LAT_SPLIT) and the latent experts (FNX_LATENT, Edge0-145B): refused on any count but two, with a plain
#     message at the load, before anything is placed. Edge0's latent split cuts D and U in two halves and its NVFP4 experts in
#     64-value blocks; four cards' FF slices are 160 wide (2.5 blocks)
# Run in the engine's source folder (the one holding src/).
import sys
from pathlib import Path

S = Path("src")
if not (S / "prefill.cu").exists():
    sys.exit("ncard4_patch.py: run it in the folder that holds src/")


def rep(name, old, new, count=1):
    p = S / name
    t = p.read_text(encoding="utf-8")
    n = t.count(old)
    if n != count:
        sys.exit(f"ncard4_patch.py: {name}: {n} matches, want {count}:\n{old[:300]}")
    p.write_text(t.replace(old, new), encoding="utf-8")


if "ncard4_patch.py" in (S / "prefill.cu").read_text(encoding="utf-8"):
    sys.exit("ncard4_patch.py: applied already")

# ---------------------------------------------------------------- prefill.cu: the gather's buffers
rep("prefill.cu",
    "    // FNX_PF_SP=1 (idea 58, sp_patch.py): see the pass's note. mixk: two chunks' mixed; ghb: host rows the other card's gather lands in\n"
    "    // (rows of D, for the draft source's R too), gnhb: the other card's; gflags [2 sets][rows in, rows read], gnflags: the other card's;\n"
    "    // ag_st: the gathers' stream; ev_hcd / ev_agd: a chunk's hc read / its gather done\n"
    "    bool sp = false;\n"
    "    float* mixk[2] = {};\n"
    "    float *ghb[2] = {}, *gnhb[2] = {};\n"
    "    unsigned *gflags = nullptr, *gnflags = nullptr;\n",
    "    // FNX_PF_SP=1 (idea 58, sp_patch.py): see the pass's note. mixk: two chunks' mixed. ncard4_patch.py: gown: host rows of this card's\n"
    "    // finished part (two sets; rows of D, for the draft source's R too), which every other card copies; gpown: every card's gown;\n"
    "    // gflags [2 sets][NCARD - 1 rows in, NCARD - 1 rows read] (slot: pf_gather_slot), gpflags: every card's;\n"
    "    // ag_st: the gathers' stream; ev_hcd / ev_agd: a chunk's hc read / its gather done\n"
    "    bool sp = false;\n"
    "    float* mixk[2] = {};\n"
    "    float* gown[2] = {};\n"
    "    float* gpown[NCARD][2] = {};\n"
    "    unsigned* gflags = nullptr;\n"
    "    unsigned* gpflags[NCARD] = {};\n")

# ---------------------------------------------------------------- prefill.cu: the gather itself
rep("prefill.cu",
    "// FNX_PF_SP (sp_patch.py): rows of base (row_floats floats a row) for a chunk of C rows on stream s: this card's part (the ring part it\n"
    "// finishes, pf_ring_plan) goes to the other card's host rows by the copy engine, the other card's part comes from this card's host rows\n"
    "// into base. Flags [set][0]: the rows are in; [set][1]: the receiver has them (the sender's next use of the set waits for that). Every\n"
    "// card makes the same calls in the same order (seq)\n"
    "void pf_gather(Engine::Pf& p, int g, float* base, int row_floats, int C, cudaStream_t s) {\n"
    "    static_assert(NCARD == 2, \"pf_gather: two cards\");\n"
    "    const unsigned seq = ++p.gseq;\n"
    "    const int set = (int) (seq & 1u), own = (g + 1) % NCARD, oth = g;\n"
    "    auto r0 = [&](int k) { return k * C / NCARD; };\n"
    "    auto rows = [&](int k) { return r0(k + 1) - r0(k); };\n"
    "    CUstream cs = (CUstream) s;\n"
    "    if (seq > 2) cud(cuStreamWaitValue32(cs, (CUdeviceptr) (p.gflags + set * 2 + 1), seq - 2, CU_STREAM_WAIT_VALUE_GEQ), \"gather consumed\");\n"
    "    if (rows(own) > 0)\n"
    "        ckc(cudaMemcpyAsync(p.gnhb[set], base + (size_t) r0(own) * row_floats, (size_t) rows(own) * row_floats * sizeof(float), cudaMemcpyDefault, s),\n"
    "            \"gather send\");\n"
    "    cud(cuStreamWriteValue32(cs, (CUdeviceptr) (p.gnflags + set * 2), seq, 0), \"gather flag\");\n"
    "    cud(cuStreamWaitValue32(cs, (CUdeviceptr) (p.gflags + set * 2), seq, CU_STREAM_WAIT_VALUE_GEQ), \"gather wait\");\n"
    "    if (rows(oth) > 0)\n"
    "        ckc(cudaMemcpyAsync(base + (size_t) r0(oth) * row_floats, p.ghb[set], (size_t) rows(oth) * row_floats * sizeof(float), cudaMemcpyDefault, s),\n"
    "            \"gather in\");\n"
    "    cud(cuStreamWriteValue32(cs, (CUdeviceptr) (p.gnflags + set * 2 + 1), seq, 0), \"gather read\");\n"
    "}\n",
    "// FNX_PF_SP (sp_patch.py; ncard4_patch.py: any count of cards): rows of base (row_floats floats a row) for a chunk of C rows on stream s.\n"
    "// This card's part (the ring part it finishes, pf_ring_plan: g + 1) goes once, by the copy engine, to its own host rows (gown), and a\n"
    "// flag tells every other card; each other card's part comes from that card's host rows into base. All host rows are in the same RAM,\n"
    "// so a part crosses the host link up once, not once a receiver. Flags on card g, [set][slot]: the rows of the card in that slot are\n"
    "// in; [set][NCARD - 1 + slot]: the card in that slot has read g's rows (g's next use of the set waits for all of them). Slots are\n"
    "// pf_gather_slot's (two cards: 0, so [set][0] and [set][1] as before). Every card makes the same calls in the same order (seq)\n"
    "void pf_gather(Engine::Pf& p, int g, float* base, int row_floats, int C, cudaStream_t s) {\n"
    "    const unsigned seq = ++p.gseq;\n"
    "    constexpr int SL = NCARD - 1, SET = 2 * SL;\n"
    "    const int set = (int) (seq & 1u), own = (g + 1) % NCARD;\n"
    "    auto r0 = [&](int k) { return k * C / NCARD; };\n"
    "    auto rows = [&](int k) { return r0(k + 1) - r0(k); };\n"
    "    CUstream cs = (CUstream) s;\n"
    "    if (seq > 2)\n"
    "        for (int j = 0; j < SL; ++j)\n"
    "            cud(cuStreamWaitValue32(cs, (CUdeviceptr) (p.gflags + set * SET + SL + j), seq - 2, CU_STREAM_WAIT_VALUE_GEQ), \"gather consumed\");\n"
    "    if (rows(own) > 0)\n"
    "        ckc(cudaMemcpyAsync(p.gown[set], base + (size_t) r0(own) * row_floats, (size_t) rows(own) * row_floats * sizeof(float), cudaMemcpyDefault, s),\n"
    "            \"gather send\");\n"
    "    for (int d = 0; d < NCARD; ++d)\n"
    "        if (d != g) cud(cuStreamWriteValue32(cs, (CUdeviceptr) (p.gpflags[d] + set * SET + pf_gather_slot(NCARD, g, d)), seq, 0), \"gather flag\");\n"
    "    for (int k = 0; k < NCARD; ++k) {\n"
    "        if (k == g) continue;\n"
    "        const int part = (k + 1) % NCARD;   // the part card k finishes\n"
    "        cud(cuStreamWaitValue32(cs, (CUdeviceptr) (p.gflags + set * SET + pf_gather_slot(NCARD, k, g)), seq, CU_STREAM_WAIT_VALUE_GEQ), \"gather wait\");\n"
    "        if (rows(part) > 0)\n"
    "            ckc(cudaMemcpyAsync(base + (size_t) r0(part) * row_floats, p.gpown[k][set], (size_t) rows(part) * row_floats * sizeof(float),\n"
    "                                cudaMemcpyDefault, s),\n"
    "                \"gather in\");\n"
    "    }\n"
    "    for (int k = 0; k < NCARD; ++k)\n"
    "        if (k != g) cud(cuStreamWriteValue32(cs, (CUdeviceptr) (p.gpflags[k] + set * SET + SL + pf_gather_slot(NCARD, g, k)), seq, 0), \"gather read\");\n"
    "}\n")

# ---------------------------------------------------------------- prefill.cu: the latent exchange stays two-card, refused at run time
rep("prefill.cu",
    "    static_assert(NCARD == 2, \"pf_lat_swap: two cards\");\n",
    "    // ncard4_patch.py: two cards only; Model refuses FNX_LATENT on any other count at the load, so this is never reached there\n"
    "    if constexpr (NCARD != 2) throw std::runtime_error(\"pf_lat_swap: FNX_LAT_SPLIT runs on two cards only\");\n")

# ---------------------------------------------------------------- prefill.cu: reset, allocation, links
rep("prefill.cu",
    "            ckc(cudaMemset(p.gflags, 0, 4 * sizeof(unsigned)), \"gather flags\");\n",
    "            ckc(cudaMemset(p.gflags, 0, 4 * (NCARD - 1) * sizeof(unsigned)), \"gather flags\");   // ncard4_patch.py\n")
rep("prefill.cu",
    "        // FNX_PF_SP (idea 58, sp_patch.py): a second mixed, the gathers' host rows (half a chunk of rows of D, two sets), flags, stream\n"
    "        if (env_int(\"FNX_PF_SP\", 0) != 0 && NCARD == 2 && p->overlap && !p->ring_sm) {\n"
    "            p->sp = true;\n"
    "            p->mixk[0] = p->mixed;\n"
    "            p->mixk[1] = palloc<float>((size_t) pf_c_ * N);\n"
    "            for (int k = 0; k < 2; ++k) p->ghb[k] = hpalloc<float>((size_t) (pf_c_ + 1) / 2 * D);\n"
    "            p->gflags = hpalloc<unsigned>(4);\n",
    "        // FNX_PF_SP (idea 58, sp_patch.py): a second mixed, the gathers' host rows (this card's part of a chunk of rows of D, two sets),\n"
    "        // flags, stream (ncard4_patch.py: any count of cards)\n"
    "        if (env_int(\"FNX_PF_SP\", 0) != 0 && p->overlap && !p->ring_sm) {\n"
    "            p->sp = true;\n"
    "            p->mixk[0] = p->mixed;\n"
    "            p->mixk[1] = palloc<float>((size_t) pf_c_ * N);\n"
    "            for (int k = 0; k < 2; ++k) p->gown[k] = hpalloc<float>((size_t) (pf_c_ + NCARD - 1) / NCARD * D);\n"
    "            p->gflags = hpalloc<unsigned>((size_t) 4 * (NCARD - 1));\n")
rep("prefill.cu",
    "    for (int c = 0; c < NCARD; ++c) {   // FNX_PF_SP: each card's gathers land in the other card's host rows\n"
    "        Pf& p = *pf_[c];\n"
    "        const Pf& n = *pf_[(c + 1) % NCARD];\n"
    "        if (p.sp != n.sp) throw std::runtime_error(\"FNX_PF_SP: on every card or on none\");\n"
    "        if (!p.sp) continue;\n"
    "        for (int k = 0; k < 2; ++k) p.gnhb[k] = n.ghb[k];\n"
    "        p.gnflags = n.gflags;\n"
    "    }\n",
    "    for (int c = 0; c < NCARD; ++c) {   // FNX_PF_SP: every card reads the others' host rows and sets their flags (ncard4_patch.py)\n"
    "        Pf& p = *pf_[c];\n"
    "        const Pf& n = *pf_[(c + 1) % NCARD];\n"
    "        if (p.sp != n.sp) throw std::runtime_error(\"FNX_PF_SP: on every card or on none\");\n"
    "        if (!p.sp) continue;\n"
    "        for (int d = 0; d < NCARD; ++d) {\n"
    "            for (int k = 0; k < 2; ++k) p.gpown[d][k] = pf_[d]->gown[k];\n"
    "            p.gpflags[d] = pf_[d]->gflags;\n"
    "        }\n"
    "    }\n")

# ---------------------------------------------------------------- model.cpp: Edge0-145B refused on four cards, at the load
rep("model.cpp",
    "    if (g.latent()) {   // edge0_patch.py\n",
    "    if (g.latent()) {   // edge0_patch.py\n"
    "        if (NCARD != 2)   // ncard4_patch.py\n"
    "            throw std::runtime_error(\"Edge0-145B (FNX_LATENT) runs on two cards only: its latent experts are cut in NVFP4's 64-value blocks \"\n"
    "                                     \"and FNX_LAT_SPLIT in two halves, and \" + std::to_string(NCARD) + \" cards' FF slices are \" +\n"
    "                                     std::to_string(FF / NCARD) + \" wide. Run Edge0 with the two-card build (bash build.sh), or \"\n"
    "                                     \"IQ3_S with this one\");\n")
print("ncard4_patch.py: pf_gather for any count of cards, Edge0 refused on more than two")
