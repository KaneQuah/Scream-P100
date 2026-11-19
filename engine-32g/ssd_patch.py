#!/usr/bin/env python3
# ssd_patch.py (2026-11-19, mikro/q4h32/; the user: "verzijo q4 za uporabnike ki imajo samo 32gb rama ... prefill iz ssd v ram in iz
# rama v vram ... vnaprej nalagal vsaj 2 lejerja"): FNX_SSD_EXPERTS=1, no routed expert in page-locked RAM (ssd.hpp says how).
# Run in the engine's source folder (the one holding src/); ssd.hpp beside this script goes to src/. Off by default: with the
# switch unset every changed line does what it did.
import shutil
import sys
from pathlib import Path

S = Path("src")
if not (S / "model.cpp").exists():
    sys.exit("ssd_patch.py: run it in the folder that holds src/")


def rep(name, old, new, count=1):
    p = S / name
    t = p.read_text(encoding="utf-8")
    n = t.count(old)
    if n != count:
        sys.exit(f"ssd_patch.py: {name}: {n} matches, want {count}:\n{old[:300]}")
    p.write_text(t.replace(old, new), encoding="utf-8")


if "ssd_patch.py" in (S / "model.cpp").read_text(encoding="utf-8"):
    sys.exit("ssd_patch.py: applied already")
shutil.copy(Path(__file__).resolve().parent / "ssd.hpp", S / "ssd.hpp")

# ---------------------------------------------------------------- model.cpp: no host copies, the file places, the tags, the fill
rep("model.cpp", '#include "model.hpp"\n', '#include "model.hpp"\n#include "ssd.hpp"   // ssd_patch.py\n')
rep("model.cpp",
    "    std::vector<void*>* track = nullptr;  // a worker placing its experts: what it allocates (Model::free_experts)\n",
    "    std::vector<void*>* track = nullptr;  // a worker placing its experts: what it allocates (Model::free_experts)\n"
    "    int layer = -1;                       // ssd_patch.py: the layer being loaded (FNX_SSD_EXPERTS notes where its experts are in the file)\n")
rep("model.cpp",
    "        std::vector<uint8_t> pt((size_t) np * stride);\n"
    "        uint8_t* h = np < n ? expert_buf((size_t) (n - np) * stride, host_of, tmp) : nullptr;\n"
    "        for (int e = 0, kh = 0, kp = 0; e < n; ++e) copy(e, np && perm[e] ? &pt[(size_t) kp++ * stride] : h + (size_t) kh++ * stride);\n",
    "        std::vector<uint8_t> pt((size_t) np * stride);\n"
    "        // ssd_patch.py, FNX_SSD_EXPERTS: no host copies; the others' bytes stay in the file (load_card tags their host_of)\n"
    "        const bool ssd = ssd_on() && !store;\n"
    "        uint8_t* h = np < n && !ssd ? expert_buf((size_t) (n - np) * stride, host_of, tmp) : nullptr;\n"
    "        for (int e = 0, kh = 0, kp = 0; e < n; ++e)\n"
    "            if (np && perm[e]) copy(e, &pt[(size_t) kp++ * stride]);\n"
    "            else if (h) copy(e, h + (size_t) kh++ * stride);\n")
rep("model.cpp",
    "                host_of[e] = dp + (size_t) kh++ * stride;\n",
    "                host_of[e] = h ? dp + (size_t) kh++ * stride : nullptr;   // ssd_patch.py: null, tagged by load_card\n")
# expert_rows2: gate and up, the card's rows
rep("model.cpp",
    "        const size_t half = (size_t) (r1 - r0) * w.row_bytes, src_mat = (size_t) t1.ne[1] * w.row_bytes;\n"
    "        experts(w, stride, (int) t1.ne[2], host_of, perm, perm_at, [&](int64_t e, uint8_t* dst) {\n",
    "        const size_t half = (size_t) (r1 - r0) * w.row_bytes, src_mat = (size_t) t1.ne[1] * w.row_bytes;\n"
    "        if (ssd_on() && host_of && !store) {   // ssd_patch.py: where the card's part of each expert is in the file\n"
    "            if (t1.ne[2] != NEXP) throw std::runtime_error(\"FNX_SSD_EXPERTS: \" + t1.name + \" does not hold NEXP experts\");\n"
    "            std::vector<SsdLayer>& v = g_ssd_layers[c.dev & (SSD_DEVS - 1)];\n"
    "            if (v.size() != (size_t) NL) v.resize(NL);\n"
    "            SsdLayer& s = v.at((size_t) layer);\n"
    "            s.fd_gate = ssd_fd(t1), s.fd_up = ssd_fd(t2), s.gate0 = t1.offset, s.up0 = t2.offset;\n"
    "            s.gu_mat = src_mat, s.gu_skip = (size_t) r0 * w.row_bytes, s.gu_half = half;\n"
    "        }\n"
    "        experts(w, stride, (int) t1.ne[2], host_of, perm, perm_at, [&](int64_t e, uint8_t* dst) {\n")
# expert_cols: down, the card's blocks of every row
rep("model.cpp",
    "        experts(w, stride, (int) t.ne[2], host_of, perm, perm_at, [&](int64_t e, uint8_t* dst) {\n"
    "            for (int r = 0; r < w.n_out; ++r)\n",
    "        if (ssd_on() && host_of && !store) {   // ssd_patch.py\n"
    "            if (t.ne[2] != NEXP) throw std::runtime_error(\"FNX_SSD_EXPERTS: \" + t.name + \" does not hold NEXP experts\");\n"
    "            std::vector<SsdLayer>& v = g_ssd_layers[c.dev & (SSD_DEVS - 1)];\n"
    "            if (v.size() != (size_t) NL) v.resize(NL);\n"
    "            SsdLayer& s = v.at((size_t) layer);\n"
    "            s.fd_down = ssd_fd(t), s.down0 = t.offset, s.dn_mat = src_mat, s.dn_row = src_row;\n"
    "            s.dn_skip = (size_t) b0 * ti.bytes, s.dn_take = w.row_bytes, s.dn_rows = w.n_out;\n"
    "        }\n"
    "        experts(w, stride, (int) t.ne[2], host_of, perm, perm_at, [&](int64_t e, uint8_t* dst) {\n"
    "            for (int r = 0; r < w.n_out; ++r)\n")
# the slots' first fill
rep("model.cpp",
    "        for (int s = 0; s < ec.n_dyn; ++s) {\n"
    "            const int i = owner[(size_t) s], p = ec.pool_of[i / NEXP];\n",
    "        std::vector<SsdFill> ssd_jobs;   // ssd_patch.py: the slots of experts with no host copy, filled from the file\n"
    "        for (int s = 0; s < ec.n_dyn; ++s) {\n"
    "            const int i = owner[(size_t) s], p = ec.pool_of[i / NEXP];\n")
rep("model.cpp",
    "            ck(cudaMemcpy(at, hgu[(size_t) i], L.gu_e_stride, cudaMemcpyDefault), \"expert cache: fill\");\n"
    "            ck(cudaMemcpy(at + ec.dn_off[p], hdn[(size_t) i], L.down_e_stride, cudaMemcpyDefault), \"expert cache: fill\");\n",
    "            if (ssd_tagged(hgu[(size_t) i])) {\n"
    "                ssd_jobs.push_back({i, at, at + ec.dn_off[p], L.gu_e_stride, L.down_e_stride});\n"
    "            } else {\n"
    "                ck(cudaMemcpy(at, hgu[(size_t) i], L.gu_e_stride, cudaMemcpyDefault), \"expert cache: fill\");\n"
    "                ck(cudaMemcpy(at + ec.dn_off[p], hdn[(size_t) i], L.down_e_stride, cudaMemcpyDefault), \"expert cache: fill\");\n"
    "            }\n")
rep("model.cpp",
    "        ec.ep_gu = put<const char*>(epg.data(), epg.size() * sizeof(char*));\n",
    "        if (!ssd_jobs.empty()) {\n"
    "            ssd_fill(c.dev, ssd_jobs, nullptr);\n"
    "            ssd_report(\"the slots filled\");\n"
    "        }\n"
    "        ec.ep_gu = put<const char*>(epg.data(), epg.size() * sizeof(char*));\n")
# load_card: the layer for the notes, Edge0 refused, the tags
rep("model.cpp",
    "    for (int l = 0; l < NL; ++l) {\n"
    "        LayerW& w = c.L[l];\n"
    "        const char** hg = host_experts ? hgu.data() + (size_t) l * NEXP : nullptr;\n",
    "    for (int l = 0; l < NL; ++l) {\n"
    "        LayerW& w = c.L[l];\n"
    "        up.layer = l;   // ssd_patch.py\n"
    "        const char** hg = host_experts ? hgu.data() + (size_t) l * NEXP : nullptr;\n")
rep("model.cpp",
    "        if (g_lat_labels) {   // edge0_patch.py: Edge0's latent experts, D and U, labels and tail scales\n",
    "        if (g_lat_labels) {   // edge0_patch.py: Edge0's latent experts, D and U, labels and tail scales\n"
    "            if (ssd_on()) throw std::runtime_error(\"FNX_SSD_EXPERTS: not for Edge0's latent experts\");   // ssd_patch.py\n")
rep("model.cpp",
    "        w.down_e = up.expert_cols(L(l, \"ffn_down_exps.weight\"), c.ff0 / 32, c.ff1 / 32, w.down_e_stride, hd, pr,\n"
    "                                  pr ? c.perm_dn.data() + (size_t) l * NEXP : nullptr);\n"
    "        }\n",
    "        w.down_e = up.expert_cols(L(l, \"ffn_down_exps.weight\"), c.ff0 / 32, c.ff1 / 32, w.down_e_stride, hd, pr,\n"
    "                                  pr ? c.perm_dn.data() + (size_t) l * NEXP : nullptr);\n"
    "        if (ssd_on() && hg && !store) {   // ssd_patch.py: the tags of the experts with no host copy (ssd.hpp)\n"
    "            const SsdLayer& s = g_ssd_layers[c.dev & (SSD_DEVS - 1)].at((size_t) l);\n"
    "            if (2 * s.gu_half != w.gu_e_stride || (size_t) s.dn_rows * s.dn_take != w.down_e_stride)\n"
    "                throw std::runtime_error(\"FNX_SSD_EXPERTS: layer \" + std::to_string(l) + \": the file places do not match the slots' layout\");\n"
    "            for (int e = 0; e < NEXP; ++e)\n"
    "                if (!(pr && pr[e])) {\n"
    "                    if (hg[e] || hd[e]) throw std::runtime_error(\"FNX_SSD_EXPERTS: an expert got a host copy\");\n"
    "                    hg[e] = ssd_tag(l * NEXP + e, false), hd[e] = ssd_tag(l * NEXP + e, true);\n"
    "                }\n"
    "        }\n"
    "        }\n")
# Model: the checks and the files, before the cards load
rep("model.cpp",
    "    const auto t0 = std::chrono::steady_clock::now();\n"
    "    Stager st(g, stage_groups(g), NCARD);\n",
    "    if (ssd_on()) {   // ssd_patch.py\n"
    "        if (how != Load::PLAIN || store) throw std::runtime_error(\"FNX_SSD_EXPERTS: a plain load only (not the lab's holder)\");\n"
    "        if (resident.empty()) throw std::runtime_error(\"FNX_SSD_EXPERTS needs the expert cache (FNX_EXPERT_RAM_GIB not off)\");\n"
    "        ssd_open(g);\n"
    "        std::fprintf(stderr, \"fnx: FNX_SSD_EXPERTS: no routed expert in page-locked RAM; read from the file through the page cache \"\n"
    "                             \"when a card needs one (%d readers, a prompt's layers %d ahead)\\n\", ssd_threads(), ssd_ahead());\n"
    "    }\n"
    "    const auto t0 = std::chrono::steady_clock::now();\n"
    "    Stager st(g, stage_groups(g), NCARD);\n")

# ---------------------------------------------------------------- engine.cu: the checks, the arena's fills
rep("engine.cu", '#include "p100_over.hpp"\n', '#include "p100_over.hpp"\n#include "ssd.hpp"   // ssd_patch.py\n')
rep("engine.cu",
    "    pf_refill_env_ = (int) env_int(\"FNX_PF_REFILL\", 0);\n",
    "    pf_refill_env_ = (int) env_int(\"FNX_PF_REFILL\", 0);\n"
    "    if (ssd_on()) {   // ssd_patch.py: every copy of an expert must go through host code that can read it from the file\n"
    "        if (pf_refill_env_) throw std::runtime_error(\"FNX_SSD_EXPERTS: FNX_PF_REFILL copies from host copies: leave it off\");\n"
    "        if (env_flag(\"FNX_PREFETCH_PROBE\")) throw std::runtime_error(\"FNX_SSD_EXPERTS: not with FNX_PREFETCH_PROBE\");\n"
    "        if (!env_flag(\"FNX_PF_STAGE_ALL\")) throw std::runtime_error(\"FNX_SSD_EXPERTS needs FNX_PF_STAGE_ALL=1 (a read stages every expert)\");\n"
    "        for (int c = 0; c < NCARD; ++c)\n"
    "            if (!cb_[(size_t) c]->dma || !cb_[(size_t) c]->st3 || !cb_[(size_t) c]->xh16)\n"
    "                throw std::runtime_error(\"FNX_SSD_EXPERTS needs the copy engine's path: FNX_P100_DMA=1 with the P100 overlap and rows\");\n"
    "    }\n")
rep("engine.cu",
    "    const bool over = live && plan_in_router && b.st3 && b.xh16 && !sh_in && (T >= over_min_t || dma);\n",
    "    const bool over = live && plan_in_router && b.st3 && b.xh16 && !sh_in && (T >= over_min_t || dma);\n"
    "    if (ssd_on() && live && !(over && dma))   // ssd_patch.py: a kernel would read host copies there are none of\n"
    "        throw std::runtime_error(\"FNX_SSD_EXPERTS: a window's misses not on the copy engine's path (FNX_SH_IN off, FNX_P100_DMA=1)\");\n")
rep("engine.cu",
    "    int* dlist = dalloc<int>(list.size());\n"
    "    ck(cudaMemcpy(dlist, list.data(), list.size() * sizeof(int), cudaMemcpyHostToDevice), \"arena: fill list\");\n"
    "    cache_copy(ec, p, dlist, st);\n",
    "    if (ssd_on()) {   // ssd_patch.py: from the file through page-locked buffers (ssd_fill), the card's device current meanwhile\n"
    "        std::vector<SsdFill> jobs;\n"
    "        for (int s = 0; s < ec.n[p]; ++s) {\n"
    "            const int i = owner[(size_t) s];\n"
    "            if (i < 0) continue;\n"
    "            const LayerW& L = w.L[(size_t) (i / NEXP)];\n"
    "            char* at = ec.pool[p] + (size_t) s * ec.slot_bytes[p];\n"
    "            jobs.push_back({i, at, at + ec.dn_off[p], L.gu_e_stride, L.down_e_stride});\n"
    "        }\n"
    "        int prev = 0;\n"
    "        ck(cudaGetDevice(&prev), \"cudaGetDevice\");\n"
    "        ck(cudaSetDevice(w.dev), \"cudaSetDevice\");\n"
    "        ssd_fill(w.dev, jobs, st);\n"
    "        ck(cudaSetDevice(prev), \"cudaSetDevice\");\n"
    "        ssd_report(\"the arena's slots filled\");\n"
    "    } else {\n"
    "    int* dlist = dalloc<int>(list.size());\n"
    "    ck(cudaMemcpy(dlist, list.data(), list.size() * sizeof(int), cudaMemcpyHostToDevice), \"arena: fill list\");\n"
    "    cache_copy(ec, p, dlist, st);\n"
    "    }\n")
rep("engine.cu",
    "    for (int k = 0; k < n; ++k) {\n"
    "        const int i = out[(size_t) k], s = to + k;\n",
    "    std::vector<SsdFill> refill;   // ssd_patch.py: the experts with no host copy, read from the file after the loop\n"
    "    for (int k = 0; k < n; ++k) {\n"
    "        const int i = out[(size_t) k], s = to + k;\n")
rep("engine.cu",
    "        ck(cudaMemcpy(at, w.ec_host_gu[(size_t) i], L.gu_e_stride, cudaMemcpyDefault), \"arena: refill\");\n"
    "        ck(cudaMemcpy(at + ec.dn_off[p], w.ec_host_dn[(size_t) i], L.down_e_stride, cudaMemcpyDefault), \"arena: refill\");\n",
    "        if (ssd_tagged(w.ec_host_gu[(size_t) i])) {\n"
    "            refill.push_back({i, at, at + ec.dn_off[p], L.gu_e_stride, L.down_e_stride});\n"
    "        } else {\n"
    "        ck(cudaMemcpy(at, w.ec_host_gu[(size_t) i], L.gu_e_stride, cudaMemcpyDefault), \"arena: refill\");\n"
    "        ck(cudaMemcpy(at + ec.dn_off[p], w.ec_host_dn[(size_t) i], L.down_e_stride, cudaMemcpyDefault), \"arena: refill\");\n"
    "        }\n")
rep("engine.cu",
    "    ck(cudaMemcpy(ec.ep_gu, epg.data(), epg.size() * sizeof(char*), cudaMemcpyHostToDevice), \"arena: tables\");\n"
    "    ck(cudaMemcpy(ec.ep_dn, epd.data(), epd.size() * sizeof(char*), cudaMemcpyHostToDevice), \"arena: tables\");\n"
    "    ck(cudaMemcpy(ec.slot_of, slot_of.data(), slot_of.size() * sizeof(int), cudaMemcpyHostToDevice), \"arena: slots\");\n",
    "    if (!refill.empty()) {   // ssd_patch.py\n"
    "        int prev = 0;\n"
    "        ck(cudaGetDevice(&prev), \"cudaGetDevice\");\n"
    "        ck(cudaSetDevice(w.dev), \"cudaSetDevice\");\n"
    "        ssd_fill(w.dev, refill, nullptr);\n"
    "        ck(cudaSetDevice(prev), \"cudaSetDevice\");\n"
    "        ssd_report(\"the arena's slots filled again\");\n"
    "    }\n"
    "    ck(cudaMemcpy(ec.ep_gu, epg.data(), epg.size() * sizeof(char*), cudaMemcpyHostToDevice), \"arena: tables\");\n"
    "    ck(cudaMemcpy(ec.ep_dn, epd.data(), epd.size() * sizeof(char*), cudaMemcpyHostToDevice), \"arena: tables\");\n"
    "    ck(cudaMemcpy(ec.slot_of, slot_of.data(), slot_of.size() * sizeof(int), cudaMemcpyHostToDevice), \"arena: slots\");\n")

# ---------------------------------------------------------------- p100_dma.cuh: the copier reads what has no host copy
rep("p100_dma.cuh", '#include "p100_over.hpp"\n', '#include "p100_over.hpp"\n#include "ssd.hpp"   // ssd_patch.py\n\n#include <condition_variable>\n')
rep("p100_dma.cuh",
    "    std::atomic<int> landed[EC_POOLS];  // each pool's count landed (DmaFill::done), read by the engine before each window\n",
    "    std::atomic<int> landed[EC_POOLS];  // each pool's count landed (DmaFill::done), read by the engine before each window\n"
    "    // ssd_patch.py, FNX_SSD_EXPERTS: the cache of experts read (FNX_SSD_RAM_GIB; null: none), else bounce buffers for box's and box2's\n"
    "    // lists (an expert a place, as listed, grown as needed); pend: each list entry's expert still to read into its place (-1: none)\n"
    "    struct SsdCache* sc = nullptr;\n"
    "    bool sc_tried = false;\n"
    "    unsigned long long sc_look = 0, sc_hit = 0;\n"
    "    char* sb[2] = {};\n"
    "    size_t sb_cap[2] = {};\n"
    "    int pend[2][DMA_MAX] = {};\n"
    "    std::vector<char> sb_tmp;\n"
    "    bool sb_said = true;   // the reads reported since the last list (ssd_report at idle)\n"
    "    struct SsdHelp* help = nullptr;   // FNX_SSD_COPIERS: the copier's readers (par[w]: they read list w's missing parts)\n"
    "    bool par[2] = {};\n")
rep("p100_dma.cuh",
    "// the card's copier: spins while lists come (decode), sleeps 100 us between looks after 20 ms without one\n",
    "// ssd_patch.py, FNX_SSD_RAM_GIB=G: a page-locked cache of G GiB a card of the experts its copier read, CLOCK. A slot is taken from no\n"
    "// list a copy may still read: the current list's (box or box2) and the other one's last (each stream synced before its list)\n"
    "struct SsdCache {\n"
    "    char* mem = nullptr;\n"
    "    size_t per = 0;              // a slot: the largest expert's part of the card, gate|up then down\n"
    "    int n = 0, hand = 0;\n"
    "    std::vector<int> owner, slot_of;\n"
    "    std::vector<unsigned char> ref;\n"
    "    std::vector<unsigned> used[2];   // the box / box2 list number that last took slot s\n"
    "    unsigned seq[2] = {1, 1};        // the current or last list number of box / box2\n"
    "};\n"
    "SsdCache* dma_ssd_cache(DmaCard* c) {\n"
    "    if (c->sc_tried) return c->sc;\n"
    "    c->sc_tried = true;\n"
    "    const double gib = ssd_ram_gib();\n"
    "    if (gib <= 0) return nullptr;\n"
    "    size_t per = 0;\n"
    "    for (const SsdLayer& s : g_ssd_layers[c->dev & (SSD_DEVS - 1)]) per = std::max(per, 2 * s.gu_half + (size_t) s.dn_rows * s.dn_take);\n"
    "    per = (per + 255) & ~(size_t) 255;\n"
    "    const long long n = per ? (long long) (gib * 1073741824.0 / (double) per) : 0;\n"
    "    if (n < 2 * DMA_MAX + 64) {\n"
    "        std::fprintf(stderr, \"fnx ssd: device %d: FNX_SSD_RAM_GIB %.2f holds %lld experts, fewer than %d: no cache\\n\", c->dev, gib, n, 2 * DMA_MAX + 64);\n"
    "        return nullptr;\n"
    "    }\n"
    "    auto* sc = new SsdCache;\n"
    "    if (cudaHostAlloc((void**) &sc->mem, (size_t) n * per, cudaHostAllocPortable) != cudaSuccess) {\n"
    "        std::fprintf(stderr, \"fnx ssd: device %d: no %.2f GiB of page-locked RAM for the cache: none\\n\", c->dev, gib);\n"
    "        cudaGetLastError();\n"
    "        delete sc;\n"
    "        return nullptr;\n"
    "    }\n"
    "    sc->per = per;\n"
    "    sc->n = (int) n;\n"
    "    sc->owner.assign((size_t) n, -1);\n"
    "    sc->slot_of.assign((size_t) NL * NEXP, -1);\n"
    "    sc->ref.assign((size_t) n, 0);\n"
    "    for (auto& u : sc->used) u.assign((size_t) n, 0u);\n"
    "    std::fprintf(stderr, \"fnx ssd: device %d: %lld experts (%.2f GiB) cached page-locked by the copier\\n\", c->dev, n, n * per / 1073741824.0);\n"
    "    return c->sc = sc;\n"
    "}\n"
    "// expert g's slot for list w: a hit, or a slot taken for it (*fresh: its bytes still to read)\n"
    "int dma_ssd_slot(SsdCache* sc, int w, int g, bool* fresh) {\n"
    "    if (const int s = sc->slot_of[(size_t) g]; s >= 0) {\n"
    "        sc->ref[(size_t) s] = 1;\n"
    "        sc->used[w][(size_t) s] = sc->seq[w];\n"
    "        *fresh = false;\n"
    "        return s;\n"
    "    }\n"
    "    for (;;) {\n"
    "        const int v = sc->hand;\n"
    "        sc->hand = (sc->hand + 1) % sc->n;\n"
    "        if (sc->used[0][(size_t) v] == sc->seq[0] || sc->used[1][(size_t) v] == sc->seq[1]) continue;\n"
    "        if (sc->ref[(size_t) v]) {\n"
    "            sc->ref[(size_t) v] = 0;\n"
    "            continue;\n"
    "        }\n"
    "        if (const int o = sc->owner[(size_t) v]; o >= 0) sc->slot_of[(size_t) o] = -1;\n"
    "        sc->owner[(size_t) v] = g;\n"
    "        sc->slot_of[(size_t) g] = v;\n"
    "        sc->ref[(size_t) v] = 1;\n"
    "        sc->used[w][(size_t) v] = sc->seq[w];\n"
    "        *fresh = true;\n"
    "        return v;\n"
    "    }\n"
    "}\n"
    "// list w's (0: box, 1: box2) tagged entries pointed at page-locked bytes, the cache's or the bounce buffer w's, once the stream s's last\n"
    "// copies have landed; c->pend[w][k]: the expert whose bytes are still to be read there (-1: none)\n"
    "void dma_ssd_list(DmaCard* c, int w, DmaEnt* e, int n, size_t gb, size_t db, cudaStream_t s) {\n"
    "    bool any = false;\n"
    "    for (int k = 0; k < n; ++k) {\n"
    "        c->pend[w][k] = -1;\n"
    "        any |= ssd_tagged(e[k].src_gu) || ssd_tagged(e[k].src_dn);\n"
    "    }\n"
    "    if (!any) return;\n"
    "    cudaStreamSynchronize(s);\n"
    "    c->sb_said = false;\n"
    "    if (SsdCache* sc = dma_ssd_cache(c); sc && gb + db <= sc->per) {\n"
    "        ++sc->seq[w];\n"
    "        for (int k = 0; k < n; ++k) {\n"
    "            if (!ssd_tagged(e[k].src_gu)) continue;\n"
    "            const int g = ssd_expert(e[k].src_gu);\n"
    "            bool fresh = false;\n"
    "            char* at = sc->mem + (size_t) dma_ssd_slot(sc, w, g, &fresh) * sc->per;\n"
    "            e[k].src_gu = at, e[k].src_dn = at + gb;\n"
    "            c->pend[w][k] = fresh ? g : -1;\n"
    "            ++c->sc_look;\n"
    "            c->sc_hit += fresh ? 0 : 1;\n"
    "        }\n"
    "        return;\n"
    "    }\n"
    "    const size_t want = (size_t) n * (gb + db);\n"
    "    if (c->sb_cap[w] < want) {\n"
    "        if (c->sb[w]) cudaFreeHost(c->sb[w]);\n"
    "        c->sb[w] = nullptr;\n"
    "        c->sb_cap[w] = 0;\n"
    "        const size_t cap = std::max(want, (size_t) 32 * (gb + db));\n"
    "        if (cudaHostAlloc((void**) &c->sb[w], cap, cudaHostAllocPortable) != cudaSuccess) {\n"
    "            std::fprintf(stderr, \"fnx ssd: the copier of device %d has no page-locked bounce buffer of %zu MiB\\n\", c->dev, cap >> 20);\n"
    "            std::abort();\n"
    "        }\n"
    "        c->sb_cap[w] = cap;\n"
    "    }\n"
    "    for (int k = 0; k < n; ++k)\n"
    "        if (ssd_tagged(e[k].src_gu)) {\n"
    "            char* at = c->sb[w] + (size_t) k * (gb + db);\n"
    "            c->pend[w][k] = ssd_expert(e[k].src_gu);\n"
    "            e[k].src_gu = at, e[k].src_dn = at + gb;\n"
    "        }\n"
    "}\n"
    "// FNX_SSD_COPIERS=n (default 4; 0: none): readers beside the card's copier that read a list's missing parts side by side, in list\n"
    "// order (an entry's gate|up, then its down), when two or more are missing; the copier issues each part's copy once its flag is up.\n"
    "// busy: the readers not yet out of the last batch (a new one starts after them)\n"
    "struct SsdHelp {\n"
    "    std::vector<std::thread> th;\n"
    "    std::mutex mu;\n"
    "    std::condition_variable cv;\n"
    "    unsigned gen = 0;\n"
    "    bool stop = false;\n"
    "    int w = 0, n = 0;\n"
    "    const DmaEnt* e = nullptr;\n"
    "    std::atomic<int> next{0}, busy{0};\n"
    "    std::atomic<int> gu_ok[DMA_MAX], dn_ok[DMA_MAX];\n"
    "};\n"
    "int ssd_copiers() {\n"
    "    static const int n = [] {\n"
    "        const char* e = std::getenv(\"FNX_SSD_COPIERS\");\n"
    "        return std::max(0, std::min(16, e ? std::atoi(e) : 4));\n"
    "    }();\n"
    "    return n;\n"
    "}\n"
    "void dma_ssd_helper(DmaCard* c, SsdHelp* s) {\n"
    "    dma_pin(c->dev);\n"
    "    std::vector<char> tmp;\n"
    "    unsigned seen = 0;\n"
    "    for (;;) {\n"
    "        {\n"
    "            std::unique_lock<std::mutex> lk(s->mu);\n"
    "            s->cv.wait(lk, [&] { return s->stop || s->gen != seen; });\n"
    "            if (s->stop) return;\n"
    "            seen = s->gen;\n"
    "        }\n"
    "        for (int k; (k = s->next.fetch_add(1)) < s->n;) {\n"
    "            const int g = c->pend[s->w][k];\n"
    "            if (g < 0) continue;\n"
    "            try {\n"
    "                ssd_read_gu(c->dev, g, const_cast<char*>(s->e[k].src_gu));\n"
    "                s->gu_ok[k].store(1, std::memory_order_release);\n"
    "                ssd_read_dn(c->dev, g, const_cast<char*>(s->e[k].src_dn), tmp);\n"
    "                s->dn_ok[k].store(1, std::memory_order_release);\n"
    "            } catch (const std::exception& x) {\n"
    "                std::fprintf(stderr, \"fnx ssd: a reader of device %d: %s\\n\", c->dev, x.what());\n"
    "                std::abort();\n"
    "            }\n"
    "        }\n"
    "        s->busy.fetch_sub(1, std::memory_order_acq_rel);\n"
    "    }\n"
    "}\n"
    "// list w's missing parts to the readers when two or more are missing (c->par[w]); else dma_ssd_read reads each itself\n"
    "void dma_ssd_go(DmaCard* c, int w, const DmaEnt* e, int n) {\n"
    "    c->par[w] = false;\n"
    "    int miss = 0;\n"
    "    for (int k = 0; k < n; ++k) miss += c->pend[w][k] >= 0 ? 1 : 0;\n"
    "    if (miss < 2 || ssd_copiers() == 0) return;\n"
    "    if (!c->help) {\n"
    "        c->help = new SsdHelp;\n"
    "        for (int i = 0; i < ssd_copiers(); ++i) c->help->th.emplace_back(dma_ssd_helper, c, c->help);\n"
    "    }\n"
    "    SsdHelp* s = c->help;\n"
    "    while (s->busy.load(std::memory_order_acquire) > 0) std::this_thread::yield();\n"
    "    for (int k = 0; k < n; ++k) {\n"
    "        s->gu_ok[k].store(0, std::memory_order_relaxed);\n"
    "        s->dn_ok[k].store(0, std::memory_order_relaxed);\n"
    "    }\n"
    "    {\n"
    "        std::lock_guard<std::mutex> g(s->mu);\n"
    "        s->w = w, s->n = n, s->e = e;\n"
    "        s->next.store(0);\n"
    "        s->busy.store((int) s->th.size());\n"
    "        ++s->gen;\n"
    "    }\n"
    "    s->cv.notify_all();\n"
    "    c->par[w] = true;\n"
    "}\n"
    "// list w's entry k's part (gate|up, or down) read into its place, when it is still to be read (by the readers: waits for it). A read\n"
    "// that fails ends the process: the card would wait for the copy forever\n"
    "void dma_ssd_read(DmaCard* c, int w, int k, const DmaEnt& e, bool dn) {\n"
    "    const int g = c->pend[w][k];\n"
    "    if (g < 0) return;\n"
    "    if (c->par[w]) {\n"
    "        const std::atomic<int>& f = dn ? c->help->dn_ok[k] : c->help->gu_ok[k];\n"
    "        while (!f.load(std::memory_order_acquire)) std::this_thread::yield();\n"
    "        return;\n"
    "    }\n"
    "    try {\n"
    "        if (dn) ssd_read_dn(c->dev, g, const_cast<char*>(e.src_dn), c->sb_tmp);\n"
    "        else ssd_read_gu(c->dev, g, const_cast<char*>(e.src_gu));\n"
    "    } catch (const std::exception& x) {\n"
    "        std::fprintf(stderr, \"fnx ssd: the copier of device %d: %s\\n\", c->dev, x.what());\n"
    "        std::abort();\n"
    "    }\n"
    "}\n"
    "// the card's copier: spins while lists come (decode), sleeps 100 us between looks after 20 ms without one\n")
rep("p100_dma.cuh",
    "                for (int k = 0; k < n2; ++k) e2[k] = c->box2->ent[k];\n"
    "                for (int k = 0; k < n2; ++k) cudaMemcpyAsync(e2[k].dst_gu, e2[k].src_gu, gb2, cudaMemcpyHostToDevice, c->cs2);   // gate|up first\n",
    "                for (int k = 0; k < n2; ++k) e2[k] = c->box2->ent[k];\n"
    "                if (ssd_on()) {   // ssd_patch.py: each part read before its copy (by the readers when two or more are missing)\n"
    "                    dma_ssd_list(c, 1, e2, n2, gb2, db2, c->cs2);\n"
    "                    dma_ssd_go(c, 1, e2, n2);\n"
    "                }\n"
    "                for (int k = 0; k < n2; ++k) {   // gate|up first\n"
    "                    if (ssd_on()) dma_ssd_read(c, 1, k, e2[k], false);\n"
    "                    cudaMemcpyAsync(e2[k].dst_gu, e2[k].src_gu, gb2, cudaMemcpyHostToDevice, c->cs2);\n"
    "                }\n")
rep("p100_dma.cuh",
    "                for (int k = 0; k < n2; ++k) cudaMemcpyAsync(e2[k].dst_dn, e2[k].src_dn, db2, cudaMemcpyHostToDevice, c->cs2);\n",
    "                for (int k = 0; k < n2; ++k) {\n"
    "                    if (ssd_on()) dma_ssd_read(c, 1, k, e2[k], true);   // ssd_patch.py\n"
    "                    cudaMemcpyAsync(e2[k].dst_dn, e2[k].src_dn, db2, cudaMemcpyHostToDevice, c->cs2);\n"
    "                }\n")
rep("p100_dma.cuh",
    "            if (std::chrono::steady_clock::now() - idle_since > std::chrono::milliseconds(20)) std::this_thread::sleep_for(std::chrono::microseconds(100));\n",
    "            if (std::chrono::steady_clock::now() - idle_since > std::chrono::milliseconds(20)) {\n"
    "                if (!c->sb_said) {   // ssd_patch.py: what a decode run read, and the cache's hits\n"
    "                    c->sb_said = true;\n"
    "                    ssd_report(\"decode\");\n"
    "                    if (c->sc_look)\n"
    "                        std::fprintf(stderr, \"fnx ssd: device %d: the cache had %llu of %llu experts asked for (%.0f %%)\\n\", c->dev, c->sc_hit, c->sc_look,\n"
    "                                     100.0 * (double) c->sc_hit / (double) c->sc_look);\n"
    "                    c->sc_look = c->sc_hit = 0;\n"
    "                }\n"
    "                std::this_thread::sleep_for(std::chrono::microseconds(100));\n"
    "            }\n")
rep("p100_dma.cuh",
    "        for (int k = 0; k < n; ++k) e[k] = c->box->ent[k];\n"
    "        DmaDev* dd = reinterpret_cast<DmaDev*>(c->d);\n"
    "        for (int k = 0; k < n; ++k) {   // gate|up first, each part's landing in gu_done[k] (split 2)\n"
    "            cudaMemcpyAsync(e[k].dst_gu, e[k].src_gu, gb, cudaMemcpyHostToDevice, c->cs);\n",
    "        for (int k = 0; k < n; ++k) e[k] = c->box->ent[k];\n"
    "        DmaDev* dd = reinterpret_cast<DmaDev*>(c->d);\n"
    "        if (ssd_on()) {   // ssd_patch.py: each part read before its copy (by the readers when two or more are missing)\n"
    "            dma_ssd_list(c, 0, e, n, gb, db, c->cs);\n"
    "            dma_ssd_go(c, 0, e, n);\n"
    "        }\n"
    "        for (int k = 0; k < n; ++k) {   // gate|up first, each part's landing in gu_done[k] (split 2)\n"
    "            if (ssd_on()) dma_ssd_read(c, 0, k, e[k], false);\n"
    "            cudaMemcpyAsync(e[k].dst_gu, e[k].src_gu, gb, cudaMemcpyHostToDevice, c->cs);\n")
rep("p100_dma.cuh",
    "        for (int k = 0; k < n; ++k) {\n"
    "            cudaMemcpyAsync(e[k].dst_dn, e[k].src_dn, db, cudaMemcpyHostToDevice, c->cs);\n",
    "        for (int k = 0; k < n; ++k) {\n"
    "            if (ssd_on()) dma_ssd_read(c, 0, k, e[k], true);   // ssd_patch.py\n"
    "            cudaMemcpyAsync(e[k].dst_dn, e[k].src_dn, db, cudaMemcpyHostToDevice, c->cs);\n")
rep("p100_dma.cuh",
    "    cudaFreeHost((void*) c->box);\n"
    "    delete c;\n",
    "    cudaFreeHost((void*) c->box);\n"
    "    for (char* b : c->sb)   // ssd_patch.py\n"
    "        if (b) cudaFreeHost(b);\n"
    "    if (c->sc) {\n"
    "        cudaFreeHost(c->sc->mem);\n"
    "        delete c->sc;\n"
    "    }\n"
    "    if (c->help) {\n"
    "        {\n"
    "            std::lock_guard<std::mutex> g(c->help->mu);\n"
    "            c->help->stop = true;\n"
    "        }\n"
    "        c->help->cv.notify_all();\n"
    "        for (auto& t : c->help->th) t.join();\n"
    "        delete c->help;\n"
    "    }\n"
    "    delete c;\n")

# ---------------------------------------------------------------- prefill.cu: every read stages, the ring, the readers ahead
rep("prefill.cu", '#include "pf_ring_plan.hpp"\n', '#include "pf_ring_plan.hpp"\n#include "ssd.hpp"   // ssd_patch.py\n')
rep("prefill.cu",
    "    cudaEvent_t ev_ready[2], ev_done[2];   // stage[k] filled; its layer's experts done with\n",
    "    cudaEvent_t ev_ready[2], ev_done[2];   // stage[k] filled; its layer's experts done with\n"
    "    // ssd_patch.py, FNX_SSD_EXPERTS (ssd.hpp): a layer's staged experts read from the file into ssd_ring[l % ssd_ring_n] (page-locked, laid out as\n"
    "    // the stage: their gate|up parts, then their down parts) ssd_ahead() layers ahead by ssd_job[l], then copied into stage[l & 1] with one\n"
    "    // copy; ssd_ev[k]: slot k's copy issued (the slot's next reader waits for it). ssd_list[l]: the experts staged at l, in order. The\n"
    "    // ring lives as long as the process\n"
    "    char* ssd_ring[8] = {};\n"
    "    size_t ssd_ring_bytes = 0;\n"
    "    int ssd_ring_n = 0;\n"
    "    cudaEvent_t ssd_ev[8] = {};\n"
    "    std::vector<int> ssd_list[NL];\n"
    "    size_t ssd_gs[NL] = {}, ssd_ds[NL] = {};\n"
    "    std::future<void> ssd_job[NL];\n"
    "    int ssd_next = 0;\n")
rep("prefill.cu",
    "    const bool staging = pf_[0]->stage[0] && (int) tokens.size() >= pf_[0]->stage_rows;\n",
    "    // ssd_patch.py, FNX_SSD_EXPERTS: every read stages (there are no host copies to read in place)\n"
    "    if (ssd_on() && !pf_[0]->stage[0]) throw std::runtime_error(\"FNX_SSD_EXPERTS: the prompt path has no stages\");\n"
    "    const bool staging = pf_[0]->stage[0] && ((int) tokens.size() >= pf_[0]->stage_rows || ssd_on());\n")
rep("prefill.cu",
    "                const int fit = std::min(n_ram, (int) (p.stage_bytes / (gs + ds)));\n",
    "                const int fit = std::min(n_ram, (int) (p.stage_bytes / (gs + ds)));\n"
    "                if (ssd_on()) {   // ssd_patch.py: every expert staged, none read in place (the last layer runs none while reading)\n"
    "                    if (fit < n_ram && l < NL - 1)\n"
    "                        throw std::runtime_error(\"FNX_SSD_EXPERTS: layer \" + std::to_string(l) + \" stages \" + std::to_string(n_ram) + \" experts, its stage holds \" +\n"
    "                                                 std::to_string(fit) + \" (FNX_PF_STAGE_MIB higher)\");\n"
    "                    p.ssd_list[l].clear();\n"
    "                    p.ssd_gs[l] = gs, p.ssd_ds[l] = ds;\n"
    "                }\n")
rep("prefill.cu",
    "                        add(w.ec_host_gu[i], gu_at + (size_t) k * gs, gs);\n"
    "                        add(w.ec_host_dn[i], dn_at + (size_t) k * ds, ds);\n",
    "                        if (ssd_on()) {\n"
    "                            p.ssd_list[l].push_back((int) i);   // ssd_patch.py: its bytes come through the ring\n"
    "                        } else {\n"
    "                        add(w.ec_host_gu[i], gu_at + (size_t) k * gs, gs);\n"
    "                        add(w.ec_host_dn[i], dn_at + (size_t) k * ds, ds);\n"
    "                        }\n")
rep("prefill.cu",
    "                        p.in_place += gs + ds;\n"
    "                    }\n"
    "                }\n"
    "            }\n",
    "                        p.in_place += gs + ds;\n"
    "                    }\n"
    "                }\n"
    "                if (ssd_on() && k) cp.push_back({nullptr, gu_at, (size_t) k * (gs + ds)});   // ssd_patch.py: the layer's one copy (src: its ring slot)\n"
    "            }\n"
    "            if (ssd_on()) {   // ssd_patch.py: a ring as large as the pass's largest layer; its readers from layer 0 again\n"
    "                size_t need = 256;\n"
    "                for (int l = 0; l < NL - 1; ++l) need = std::max(need, p.ssd_list[l].size() * (p.ssd_gs[l] + p.ssd_ds[l]));\n"
    "                const int rn = ssd_ahead() + 1;\n"
    "                if (need > p.ssd_ring_bytes || rn != p.ssd_ring_n) {\n"
    "                    for (int k = 0; k < p.ssd_ring_n; ++k) {\n"
    "                        cudaEventSynchronize(p.ssd_ev[k]);\n"
    "                        cudaFreeHost(p.ssd_ring[k]);\n"
    "                        p.ssd_ring[k] = nullptr;\n"
    "                    }\n"
    "                    p.ssd_ring_bytes = std::max(need, p.ssd_ring_bytes);\n"
    "                    for (int k = 0; k < rn; ++k) {\n"
    "                        ckc(cudaHostAlloc((void**) &p.ssd_ring[k], p.ssd_ring_bytes, cudaHostAllocPortable), \"ssd ring\");\n"
    "                        if (!p.ssd_ev[k]) ckc(cudaEventCreateWithFlags(&p.ssd_ev[k], cudaEventDisableTiming), \"ssd ring event\");\n"
    "                    }\n"
    "                    p.ssd_ring_n = rn;\n"
    "                    std::fprintf(stderr, \"fnx ssd: device %d: a prompt ring of %d x %zu MiB page-locked\\n\", w.dev, rn, p.ssd_ring_bytes >> 20);\n"
    "                }\n"
    "                p.ssd_next = 0;\n"
    "            }\n")
rep("prefill.cu",
    "    auto stage_in = [&](int l) {\n"
    "        each([&](int, CardBuf&, const CardW&, Pf& p) {\n"
    "            ckc(cudaStreamWaitEvent(p.cs, p.ev_done[l & 1], 0), \"stage wait\");\n",
    "    auto stage_in = [&](int l) {\n"
    "        each([&](int, CardBuf&, const CardW& w, Pf& p) {\n"
    "            if (ssd_on()) {   // ssd_patch.py: the readers up to ssd_ahead() layers ahead, then this layer's ring slot in with one copy\n"
    "                for (const int last = std::min(l + ssd_ahead(), NL - 2); p.ssd_next <= last; ++p.ssd_next) {\n"
    "                    const int m = p.ssd_next, slot = m % p.ssd_ring_n, dev = w.dev;\n"
    "                    Pf* pp = &p;\n"
    "                    p.ssd_job[m] = std::async(std::launch::async, [pp, m, slot, dev] {\n"
    "                        cudaSetDevice(dev);\n"
    "                        if (cudaEventSynchronize(pp->ssd_ev[slot]) != cudaSuccess) throw std::runtime_error(\"FNX_SSD_EXPERTS: a ring slot's copy\");\n"
    "                        const std::vector<int>& list = pp->ssd_list[m];\n"
    "                        const size_t gs = pp->ssd_gs[m], ds = pp->ssd_ds[m], n = list.size();\n"
    "                        char* base = pp->ssd_ring[slot];\n"
    "                        ssd_parallel(n, ssd_threads(), [&](size_t k, std::vector<char>& tmp) {\n"
    "                            ssd_read_gu(dev, list[k], base + k * gs);\n"
    "                            ssd_read_dn(dev, list[k], base + n * gs + k * ds, tmp);\n"
    "                        });\n"
    "                    });\n"
    "                }\n"
    "                if (p.ssd_job[l].valid()) p.ssd_job[l].get();\n"
    "                ckc(cudaStreamWaitEvent(p.cs, p.ev_done[l & 1], 0), \"stage wait\");\n"
    "                for (const Pf::Copy& c : p.copies[l]) ckc(cudaMemcpyAsync(c.dst, p.ssd_ring[l % p.ssd_ring_n], c.n, cudaMemcpyHostToDevice, p.cs), \"stage copy\");\n"
    "                ckc(cudaEventRecord(p.ssd_ev[l % p.ssd_ring_n], p.cs), \"ring copied\");\n"
    "                ckc(cudaEventRecord(p.ev_ready[l & 1], p.cs), \"stage ready\");\n"
    "                return;\n"
    "            }\n"
    "            ckc(cudaStreamWaitEvent(p.cs, p.ev_done[l & 1], 0), \"stage wait\");\n")
rep("prefill.cu",
    "            ckc(cudaEventRecord(p.ev_ready[l & 1], p.cs), \"stage ready\");\n"
    "        });\n"
    "    };\n",
    "            ckc(cudaEventRecord(p.ev_ready[l & 1], p.cs), \"stage ready\");\n"
    "        });\n"
    "        if (ssd_on() && l == NL - 2) ssd_report(\"a prompt pass\");   // ssd_patch.py\n"
    "    };\n")
print("ssd_patch.py: FNX_SSD_EXPERTS in model.cpp, engine.cu, p100_dma.cuh, prefill.cu; src/ssd.hpp")
