#!/usr/bin/env python3
# ssd3_patch.py (2026-11-19, mikro/q4h32/; after ssd_patch.py): v3 of FNX_SSD_EXPERTS, RAM as a real second tier for a 32 GB PC. v1/v2
# under a 27-28G memory limit read most misses from the disk (q4h32a d, q4h32b a: ~16 tok/s), since the page-locked cache was filled
# only by misses and the prompt path read every expert out of a slot from the file again. With FNX_SSD_RAM_GIB set:
#   - the copier's RAM tier is exclusive with the card's slots: the expert a list's entry evicts (ExpertCache::prev, written where a plan
#     replaces a slot's owner; DmaBox::prev) is copied down into it before the slot is written, an expert copied up from it is let go
#   - it is filled at the load with the hottest experts out of a slot (p100_dma_ssd_warm), in classes by an expert's size on the card
#   - the prompt slots' experts go down into it before a read lends their slots (Engine::pf_tails)
#   - a read stages from it what it holds (p100_dma_ssd_find); only the rest goes through the ring from the file
# v3d (13:30): a list's down copies wait for the other copy stream's last list on the card (an event), not the copier for that stream.
# v3e (14:00, for the exact UD-Q4_K_XL on 64 GB: q4h32i at 24 GiB a card read ~450 experts a request from the disk in decode, and the
# first pass after the load evicted the hottest experts of the tier):
#   - the tier is sized at the load's warm fill by need: each class its experts out of a slot and 64 more, all scaled down alike when
#     FNX_SSD_RAM_GIB does not hold that (a cap: what is not needed is not taken); no class too small for a tier any more
#   - CLOCK gives up after two sweeps (every slot protected): the evicted expert is not kept, a read goes through the bounce buffer
#   - the warm fill puts the coldest into the first slots, unmarked, so CLOCK (from slot 0) lets them go first; a prompt pass's reads
#     from the tier mark nothing
# Run in the engine's source folder after ssd_patch.py. Off (FNX_SSD_EXPERTS or FNX_SSD_RAM_GIB unset): ExpertCache::prev stays null and
# every changed line does what it did.
import sys
from pathlib import Path

S = Path("src")
if "ssd_patch.py" not in (S / "model.cpp").read_text(encoding="utf-8"):
    sys.exit("ssd3_patch.py: ssd_patch.py first")
if "ssd3_patch.py" in (S / "p100_dma.cuh").read_text(encoding="utf-8"):
    sys.exit("ssd3_patch.py: applied already")


def rep(name, old, new, count=1):
    p = S / name
    t = p.read_text(encoding="utf-8")
    n = t.count(old)
    if n != count:
        sys.exit(f"ssd3_patch.py: {name}: {n} matches, want {count}:\n{old[:300]}")
    p.write_text(t.replace(old, new), encoding="utf-8")


def between(name, start, end, new):
    """replaces [start, end) (start kept out too: new begins where start began)"""
    p = S / name
    t = p.read_text(encoding="utf-8")
    a = t.find(start)
    b = t.find(end, a + 1)
    if a < 0 or b < 0 or t.count(start) != 1:
        sys.exit(f"ssd3_patch.py: {name}: markers not found once: {start[:80]!r} / {end[:80]!r}")
    p.write_text(t[:a] + new + t[b:], encoding="utf-8")


# ---------------------------------------------------------------- ExpertCache::prev, the mailbox's prev
rep("model.hpp",
    "    float* prior = nullptr;   // FNX_ROUTE_PRIOR: lambda for the experts in a slot (k_router_route), 0 off\n",
    "    float* prior = nullptr;   // FNX_ROUTE_PRIOR: lambda for the experts in a slot (k_router_route), 0 off\n"
    "    int* prev = nullptr;      // ssd3_patch.py: [n_slots], the expert a plan last evicted from slot s (-1: none); null: not kept\n")
rep("model.cpp",
    "        ec.prior = put<float>(&noprior, sizeof noprior);\n",
    "        ec.prior = put<float>(&noprior, sizeof noprior);\n"
    "        if (ssd_on() && ssd_ram_gib() > 0) {   // ssd3_patch.py: what each plan evicts, for the copier's RAM tier\n"
    "            const std::vector<int> pv((size_t) ec.n_slots, -1);\n"
    "            ec.prev = put<int>(pv.data(), pv.size() * sizeof(int));\n"
    "        }\n")
rep("model.cpp",
    "static_assert(sizeof(CardW) == 1008, \"CardW changed: put what load_card stores in card_fields, then update this size\");\n",
    "// 1016: ssd3_patch.py's ExpertCache::prev (the ExpertCache is the worker's: Model::place_experts makes it, not card_fields)\n"
    "static_assert(sizeof(CardW) == 1016, \"CardW changed: put what load_card stores in card_fields, then update this size\");\n")
rep("p100_dma_box.cuh",
    "    DmaEnt ent[DMA_MAX];\n",
    "    DmaEnt ent[DMA_MAX];\n"
    "    int prev[DMA_MAX];   // ssd3_patch.py: the expert each entry's slot held before (ExpertCache::prev), -1: none\n")
# the plans: cache_plan's two ways and the prefetch's
rep("moe.cuh",
    "                const int g = mlist[k + rank];\n"
    "                if (const int o = ec.owner[idx]; o >= 0) {\n",
    "                const int g = mlist[k + rank];\n"
    "                if (ec.prev) ec.prev[idx] = ec.owner[idx];   // ssd3_patch.py\n"
    "                if (const int o = ec.owner[idx]; o >= 0) {\n")
rep("moe.cuh",
    "                if (lane == 0) {\n"
    "                    if (const int o = ec.owner[victim]; o >= 0) {\n",
    "                if (lane == 0) {\n"
    "                    if (ec.prev) ec.prev[victim] = ec.owner[victim];   // ssd3_patch.py\n"
    "                    if (const int o = ec.owner[victim]; o >= 0) {\n")
rep("p100_moe_rows.cuh",
    "            if (lane == 0) {\n"
    "                if (const int o = ec.owner[victim]; o >= 0) {\n",
    "            if (lane == 0) {\n"
    "                if (ec.prev) ec.prev[victim] = ec.owner[victim];   // ssd3_patch.py\n"
    "                if (const int o = ec.owner[victim]; o >= 0) {\n")
# the publishes: k_dma_publish, dma_publish2 and FNX_FOLD's
rep("p100_dma.cuh",
    "        box->ent[k] = DmaEnt{ec.host_gu[g], at, ec.host_dn[g], at + ec.dn_off[p]};\n",
    "        box->ent[k] = DmaEnt{ec.host_gu[g], at, ec.host_dn[g], at + ec.dn_off[p]};\n"
    "        box->prev[k] = ec.prev ? ec.prev[ec.slot_of[g]] : -1;   // ssd3_patch.py\n", count=2)
rep("moe.cuh",
    "            box->ent[q] = DmaEnt{ec.host_gu[g], at, ec.host_dn[g], at + ec.dn_off[p]};\n",
    "            box->ent[q] = DmaEnt{ec.host_gu[g], at, ec.host_dn[g], at + ec.dn_off[p]};\n"
    "            box->prev[q] = ec.prev ? ec.prev[ec.slot_of[g]] : -1;   // ssd3_patch.py\n")

# ---------------------------------------------------------------- the copier's RAM tier
rep("p100_dma.cuh",
    "    bool par[2] = {};\n"
    "};\n",
    "    bool par[2] = {};\n"
    "    std::mutex sc_mu;                    // ssd3_patch.py: the RAM tier's making (the copier or the engine's thread)\n"
    "    char* sv[2][DMA_MAX] = {};           // where list w's entry k's evicted expert comes down to (null: nothing to keep)\n"
    "    cudaStream_t csd[2] = {};            // list w's down copies (the copy engine of that direction), an event each (evd)\n"
    "    cudaEvent_t evd[2][DMA_MAX] = {};\n"
    "    cudaEvent_t evl[2] = {};             // each stream's last list's copies (cs, cs2): a list's down copies wait for the other's on the card\n"
    "    unsigned long long sc_kept = 0;      // evicted experts kept (down, or there already) since the last report\n"
    "};\n")
between("p100_dma.cuh",
        "// ssd_patch.py, FNX_SSD_RAM_GIB=G: a page-locked cache of G GiB a card of the experts its copier read, CLOCK.",
        "// FNX_SSD_COPIERS=n (default 4; 0: none)",
        r'''// ssd3_patch.py, FNX_SSD_RAM_GIB=G (with FNX_SSD_EXPERTS): the card's RAM tier, up to G GiB of page-locked slots in classes by an
// expert's bytes on the card (the pools' sizes), CLOCK in each. Sized at the load's warm fill by need: each class its experts out of a slot
// and SSD_SPARE_NEED more, all scaled down alike when G GiB do not hold that (made first by a list instead: a class's share by its layers'
// bytes). Exclusive with the card's slots: an expert copied up from it is let go once its copy has landed (at its stream's next list),
// and the expert a list's entry evicts from a card slot (DmaBox::prev) comes down into it before the slot is written (sv). Filled at the
// load with the hottest experts out of a slot (p100_dma_ssd_warm), given the prompt slots' experts before a read lends them
// (p100_dma_ssd_save), read by the prompt path (p100_dma_ssd_find). A slot is taken from no list a copy may still read: not the current
// list's (box or box2) nor the other stream's last (each stream is synced before its list); when CLOCK finds none in two sweeps, the
// evicted expert is not kept and a read goes through the bounce buffer. Without it (G 0): a bounce buffer per box, nothing kept.
struct SsdClass {
    size_t per = 0;
    char* mem = nullptr;
    int n = 0, hand = 0, held = 0;
    std::vector<int> owner;
    std::vector<unsigned char> ref;
    std::vector<unsigned> used[2];   // the box / box2 list number that last took slot s
    std::vector<int> free;           // the empty slots: taken before CLOCK evicts anything (it would evict a held expert first)
};
struct SsdCache {
    std::mutex mu;
    std::vector<SsdClass> cls;
    std::vector<int> cls_of;      // [NL]: each layer's class
    std::vector<int> slot_of;     // [NL * NEXP]: the expert's slot in its layer's class, -1: not here
    unsigned seq[2] = {1, 1};     // the current or last list number of box / box2
    std::vector<int> rel[2];      // experts copied up by stream w's last list: let go at its next
};
constexpr int SSD_SPARE = 2 * DMA_MAX + 64;   // a class's slots beyond what two lists can protect (the shares by the layers' bytes)
constexpr int SSD_SPARE_NEED = 64;            // sized by need: a class's slots beyond its experts out of a slot
// out: the experts out of a slot (the warm fill's), to size the tier by need; null: the shares by the layers' bytes
SsdCache* dma_ssd_cache(DmaCard* c, const std::vector<int>* out = nullptr) {
    std::lock_guard<std::mutex> lk(c->sc_mu);
    if (c->sc_tried) return c->sc;
    c->sc_tried = true;
    const double gib = ssd_ram_gib(), cap = gib * 1073741824.0;
    const std::vector<SsdLayer>& L = g_ssd_layers[c->dev & (SSD_DEVS - 1)];
    if (gib <= 0 || L.size() != (size_t) NL) return nullptr;
    auto* sc = new SsdCache;
    sc->cls_of.assign(NL, 0);
    std::vector<double> wt;
    for (int l = 0; l < NL; ++l) {
        const size_t per = (2 * L[(size_t) l].gu_half + (size_t) L[(size_t) l].dn_rows * L[(size_t) l].dn_take + 255) & ~(size_t) 255;
        size_t k = 0;
        while (k < sc->cls.size() && sc->cls[k].per != per) ++k;
        if (k == sc->cls.size()) {
            sc->cls.emplace_back();
            sc->cls.back().per = per;
            wt.push_back(0.0);
        }
        sc->cls_of[(size_t) l] = (int) k;
        wt[k] += (double) per;
    }
    std::string how;
    if (out) {   // by need: each class its experts out of a slot and SSD_SPARE_NEED more, scaled down alike when G GiB do not hold that
        std::vector<double> need(sc->cls.size(), 0.0);
        for (const int g : *out) need[(size_t) sc->cls_of[(size_t) (g / NEXP)]] += 1.0;
        double fixed = 0, var = 0;
        for (size_t k = 0; k < sc->cls.size(); ++k)
            if (need[k] > 0) {
                fixed += SSD_SPARE_NEED * (double) sc->cls[k].per;
                var += need[k] * (double) sc->cls[k].per;
            }
        const double f = var > 0 ? std::min(1.0, std::max(0.0, (cap - fixed) / var)) : 0.0;
        for (size_t k = 0; k < sc->cls.size(); ++k) sc->cls[k].n = need[k] > 0 && f > 0 ? (int) (f * need[k]) + SSD_SPARE_NEED : 0;
        char t[96];
        std::snprintf(t, sizeof t, "by need: %.1f %% of the %zu experts out of a slot", 100.0 * f, out->size());
        how = t;
    } else {
        // a class whose share holds fewer than SSD_SPARE + 64 slots (a layer or two of another size: their experts sit on the cards
        // anyway) has none, its share going to the others; its layers' lists take the bounce buffers
        for (bool again = true; again;) {
            again = false;
            double live = 0;
            for (size_t k = 0; k < sc->cls.size(); ++k) live += wt[k];
            for (size_t k = 0; k < sc->cls.size(); ++k)
                if (wt[k] > 0 && cap * wt[k] / live / (double) sc->cls[k].per < SSD_SPARE + 64) {
                    wt[k] = 0;
                    again = true;
                }
        }
        double live = 0;
        for (double x : wt) live += x;
        for (size_t k = 0; k < sc->cls.size(); ++k) sc->cls[k].n = wt[k] > 0 && live > 0 ? (int) (cap * wt[k] / live / (double) sc->cls[k].per) : 0;
        how = "by the layers' bytes";
    }
    std::string say;
    bool ok = false;
    size_t bytes = 0;
    for (const SsdClass& s : sc->cls) ok |= s.n > 0;
    for (size_t k = 0; k < sc->cls.size() && ok; ++k) {
        SsdClass& s = sc->cls[k];
        if (s.n == 0) {
            say += (k ? ", " : "") + std::string("none of ") + std::to_string(s.per >> 10) + " KiB";
            continue;
        }
        if (cudaHostAlloc((void**) &s.mem, (size_t) s.n * s.per, cudaHostAllocPortable) != cudaSuccess) {
            cudaGetLastError();
            ok = false;
            break;
        }
        s.owner.assign((size_t) s.n, -1);
        s.ref.assign((size_t) s.n, 0);
        for (auto& u : s.used) u.assign((size_t) s.n, 0u);
        s.free.reserve((size_t) s.n);
        for (int v = s.n - 1; v >= 0; --v) s.free.push_back(v);
        bytes += (size_t) s.n * s.per;
        say += (k ? ", " : "") + std::to_string(s.n) + " x " + std::to_string(s.per >> 10) + " KiB";
    }
    if (!ok) {
        for (SsdClass& s : sc->cls)
            if (s.mem) cudaFreeHost(s.mem);
        delete sc;
        std::fprintf(stderr, "fnx ssd: device %d: no RAM tier of %.2f GiB (no page-locked RAM, or no class has slots): bounce buffers only\n",
                     c->dev, gib);
        return nullptr;
    }
    sc->slot_of.assign((size_t) NL * NEXP, -1);
    std::fprintf(stderr, "fnx ssd: device %d: a RAM tier of %.2f GiB page-locked (at most %.2f), %s (%s)\n", c->dev, bytes / 1073741824.0, gib,
                 how.c_str(), say.c_str());
    return c->sc = sc;
}
// (the callers hold sc->mu) whether g's class has slots
bool sc_on(const SsdCache* sc, int g) { return sc->cls[(size_t) sc->cls_of[(size_t) (g / NEXP)]].n > 0; }
// g's bytes in its slot
char* sc_at(SsdCache* sc, int g) {
    const SsdClass& s = sc->cls[(size_t) sc->cls_of[(size_t) (g / NEXP)]];
    return s.mem + (size_t) sc->slot_of[(size_t) g] * s.per;
}
// a slot of g's class taken for g: an empty one while there are, else CLOCK (none a list may still read); w: the list (0 box, 1 box2)
// that writes or reads it now, -1 none. Null when two sweeps find none (every slot protected: a small class)
char* sc_take(SsdCache* sc, int w, int g) {
    SsdClass& s = sc->cls[(size_t) sc->cls_of[(size_t) (g / NEXP)]];
    while (!s.free.empty()) {
        const int v = s.free.back();
        s.free.pop_back();
        if (s.owner[(size_t) v] >= 0) continue;   // taken by CLOCK since it was let go
        ++s.held;
        s.owner[(size_t) v] = g;
        sc->slot_of[(size_t) g] = v;
        s.ref[(size_t) v] = 1;
        if (w >= 0) s.used[w][(size_t) v] = sc->seq[w];
        return s.mem + (size_t) v * s.per;
    }
    for (int step = 0; step < 2 * s.n; ++step) {
        const int v = s.hand;
        s.hand = (s.hand + 1) % s.n;
        if (s.used[0][(size_t) v] == sc->seq[0] || s.used[1][(size_t) v] == sc->seq[1]) continue;
        if (s.ref[(size_t) v]) {
            s.ref[(size_t) v] = 0;
            continue;
        }
        if (const int o = s.owner[(size_t) v]; o >= 0) sc->slot_of[(size_t) o] = -1;
        else ++s.held;
        s.owner[(size_t) v] = g;
        sc->slot_of[(size_t) g] = v;
        s.ref[(size_t) v] = 1;
        if (w >= 0) s.used[w][(size_t) v] = sc->seq[w];
        return s.mem + (size_t) v * s.per;
    }
    return nullptr;
}
// g, which is here, used now by list w (-1: by nobody now); a release still pending for it is dropped (it is wanted here again)
char* sc_hit(SsdCache* sc, int w, int g) {
    SsdClass& s = sc->cls[(size_t) sc->cls_of[(size_t) (g / NEXP)]];
    const int v = sc->slot_of[(size_t) g];
    s.ref[(size_t) v] = 1;
    if (w >= 0) s.used[w][(size_t) v] = sc->seq[w];
    for (auto& r : sc->rel) r.erase(std::remove(r.begin(), r.end(), g), r.end());
    return sc_at(sc, g);
}
void sc_drop(SsdCache* sc, int g) {
    const int v = sc->slot_of[(size_t) g];
    if (v < 0) return;
    SsdClass& s = sc->cls[(size_t) sc->cls_of[(size_t) (g / NEXP)]];
    s.owner[(size_t) v] = -1;
    s.ref[(size_t) v] = 0;
    --s.held;
    sc->slot_of[(size_t) g] = -1;
    s.free.push_back(v);
}
// list w's bounce buffer, n entries of gb + db bytes (stream w synced: no copy reads it)
char* dma_ssd_bounce(DmaCard* c, int w, int n, size_t gb, size_t db) {
    const size_t want = (size_t) n * (gb + db);
    if (c->sb_cap[w] < want) {
        if (c->sb[w]) cudaFreeHost(c->sb[w]);
        c->sb[w] = nullptr;
        c->sb_cap[w] = 0;
        const size_t cap = std::max(want, (size_t) 32 * (gb + db));
        if (cudaHostAlloc((void**) &c->sb[w], cap, cudaHostAllocPortable) != cudaSuccess) {
            std::fprintf(stderr, "fnx ssd: the copier of device %d has no page-locked bounce buffer of %zu MiB\n", c->dev, cap >> 20);
            std::abort();
        }
        c->sb_cap[w] = cap;
    }
    return c->sb[w];
}
// list w (0: box, 1: box2), once stream s's last copies have landed: each entry's evicted expert (prev) kept in the RAM tier (c->sv[w][k]:
// where it comes down to, null when it is there already or no slot is free) and each tagged entry pointed at page-locked bytes, the
// tier's (a hit, or a slot it is read into: c->pend[w][k]) or the bounce buffer w's (no tier, or no slot)
void dma_ssd_list(DmaCard* c, int w, DmaEnt* e, const int* prev, int n, size_t gb, size_t db, cudaStream_t s) {
    SsdCache* sc = ssd_ram_gib() > 0 ? dma_ssd_cache(c) : nullptr;
    int gl = -1;   // an expert of the list's layer (a list is one layer's: one class)
    for (int k = 0; k < n && gl < 0; ++k) gl = ssd_tagged(e[k].src_gu) ? ssd_expert(e[k].src_gu) : prev[k];
    if (sc && (gl < 0 || !sc_on(sc, gl))) sc = nullptr;   // a class with no slots: the bounce buffer
    bool any = false;
    for (int k = 0; k < n; ++k) {
        c->pend[w][k] = -1;
        c->sv[w][k] = nullptr;
        any |= ssd_tagged(e[k].src_gu) || ssd_tagged(e[k].src_dn) || (sc && prev[k] >= 0);
    }
    if (!any) return;
    cudaStreamSynchronize(s);
    c->sb_said = false;
    if (sc) {
        std::lock_guard<std::mutex> lk(sc->mu);
        for (const int g : sc->rel[w]) sc_drop(sc, g);
        sc->rel[w].clear();
        ++sc->seq[w];
        for (int k = 0; k < n; ++k) {
            if (const int o = prev[k]; o >= 0) {   // down before the slot is written, unless it is here already (or no slot is free)
                if (sc->slot_of[(size_t) o] >= 0) {
                    sc_hit(sc, w, o);
                    ++c->sc_kept;
                } else if ((c->sv[w][k] = sc_take(sc, w, o)) != nullptr)
                    ++c->sc_kept;
            }
            if (!ssd_tagged(e[k].src_gu)) continue;
            const int g = ssd_expert(e[k].src_gu);
            char* at;
            ++c->sc_look;
            if (sc->slot_of[(size_t) g] >= 0) {
                at = sc_hit(sc, w, g);
                ++c->sc_hit;
            } else {
                at = sc_take(sc, w, g);
                c->pend[w][k] = g;
                if (!at) {   // no slot: read through the bounce buffer, not kept
                    at = dma_ssd_bounce(c, w, n, gb, db) + (size_t) k * (gb + db);
                    e[k].src_gu = at, e[k].src_dn = at + gb;
                    continue;
                }
            }
            sc->rel[w].push_back(g);   // it lives on the card now: let go once this copy has landed
            e[k].src_gu = at, e[k].src_dn = at + gb;
        }
        return;
    }
    char* sb = dma_ssd_bounce(c, w, n, gb, db);
    for (int k = 0; k < n; ++k)
        if (ssd_tagged(e[k].src_gu)) {
            char* at = sb + (size_t) k * (gb + db);
            c->pend[w][k] = ssd_expert(e[k].src_gu);
            e[k].src_gu = at, e[k].src_dn = at + gb;
        }
}
// list w's evicted experts (c->sv) copied down on the down stream w, an event each: that direction's copy engine takes them while the
// other copies the list's experts up, each up copy waiting for its own slot's down copy only
void dma_ssd_down(DmaCard* c, int w, const DmaEnt* e, int n, size_t gb, size_t db) {
    bool any = false;
    for (int k = 0; k < n && !any; ++k) any = c->sv[w][k] != nullptr;
    if (!any) return;
    if (!c->csd[w]) {
        bool ok = cudaStreamCreateWithFlags(&c->csd[w], cudaStreamNonBlocking) == cudaSuccess;
        for (int k = 0; k < DMA_MAX && ok; ++k) ok = cudaEventCreateWithFlags(&c->evd[w][k], cudaEventDisableTiming) == cudaSuccess;
        if (!ok) {
            std::fprintf(stderr, "fnx ssd: the copier of device %d: no down stream: %s\n", c->dev, cudaGetErrorString(cudaGetLastError()));
            std::abort();
        }
    }
    // a slot the other stream's last list copied up has landed before it is copied down (on the card: the copier does not wait)
    if (c->evl[1 - w]) cudaStreamWaitEvent(c->csd[w], c->evl[1 - w], 0);
    for (int k = 0; k < n; ++k)
        if (char* d = c->sv[w][k]) {
            cudaMemcpyAsync(d, e[k].dst_gu, gb, cudaMemcpyDeviceToHost, c->csd[w]);
            cudaMemcpyAsync(d + gb, e[k].dst_dn, db, cudaMemcpyDeviceToHost, c->csd[w]);
            cudaEventRecord(c->evd[w][k], c->csd[w]);
        }
}
// stream s's (list w's) copies issued so far, for the other stream's next down copies
void dma_ssd_mark(DmaCard* c, int w, cudaStream_t s) {
    if (!c->sc) return;   // no RAM tier: nothing is copied down
    if (!c->evl[w] && cudaEventCreateWithFlags(&c->evl[w], cudaEventDisableTiming) != cudaSuccess) {
        std::fprintf(stderr, "fnx ssd: the copier of device %d: no list event: %s\n", c->dev, cudaGetErrorString(cudaGetLastError()));
        std::abort();
    }
    cudaEventRecord(c->evl[w], s);
}
''')
rep("p100_dma.cuh",
    "                cuStreamWriteValue32((CUstream) c->cs2, (CUdeviceptr) &reinterpret_cast<DmaDev*>(c->d)->flag2, s2, 0);\n",
    "                cuStreamWriteValue32((CUstream) c->cs2, (CUdeviceptr) &reinterpret_cast<DmaDev*>(c->d)->flag2, s2, 0);\n"
    "                if (ssd_on()) dma_ssd_mark(c, 1, c->cs2);   // ssd3_patch.py\n")
rep("p100_dma.cuh",
    "        cuStreamWriteValue32((CUstream) c->cs, (CUdeviceptr) &reinterpret_cast<DmaDev*>(c->d)->flag, s, 0);\n",
    "        cuStreamWriteValue32((CUstream) c->cs, (CUdeviceptr) &reinterpret_cast<DmaDev*>(c->d)->flag, s, 0);\n"
    "        if (ssd_on()) dma_ssd_mark(c, 0, c->cs);   // ssd3_patch.py\n")
# the loop: prev to the lists, the evicted expert down before its slot is written
rep("p100_dma.cuh",
    "                    dma_ssd_list(c, 1, e2, n2, gb2, db2, c->cs2);\n",
    "                    int pv2[DMA_MAX];   // ssd3_patch.py\n"
    "                    for (int k = 0; k < n2; ++k) pv2[k] = c->box2->prev[k];\n"
    "                    dma_ssd_list(c, 1, e2, pv2, n2, gb2, db2, c->cs2);\n")
rep("p100_dma.cuh",
    "                for (int k = 0; k < n2; ++k) {   // gate|up first\n"
    "                    if (ssd_on()) dma_ssd_read(c, 1, k, e2[k], false);\n",
    "                if (ssd_on()) dma_ssd_down(c, 1, e2, n2, gb2, db2);   // ssd3_patch.py: the slots' last experts down into the RAM tier\n"
    "                for (int k = 0; k < n2; ++k) {   // gate|up first\n"
    "                    if (ssd_on() && c->sv[1][k]) cudaStreamWaitEvent(c->cs2, c->evd[1][k], 0);   // ssd3_patch.py: once its last expert is down\n"
    "                    if (ssd_on()) dma_ssd_read(c, 1, k, e2[k], false);\n")
rep("p100_dma.cuh",
    "            dma_ssd_list(c, 0, e, n, gb, db, c->cs);\n",
    "            int pv[DMA_MAX];   // ssd3_patch.py\n"
    "            for (int k = 0; k < n; ++k) pv[k] = c->box->prev[k];\n"
    "            dma_ssd_list(c, 0, e, pv, n, gb, db, c->cs);\n")
rep("p100_dma.cuh",
    "        for (int k = 0; k < n; ++k) {   // gate|up first, each part's landing in gu_done[k] (split 2)\n"
    "            if (ssd_on()) dma_ssd_read(c, 0, k, e[k], false);\n",
    "        if (ssd_on()) dma_ssd_down(c, 0, e, n, gb, db);   // ssd3_patch.py: the slots' last experts down into the RAM tier\n"
    "        for (int k = 0; k < n; ++k) {   // gate|up first, each part's landing in gu_done[k] (split 2)\n"
    "            if (ssd_on() && c->sv[0][k]) cudaStreamWaitEvent(c->cs, c->evd[0][k], 0);   // ssd3_patch.py: once its last expert is down\n"
    "            if (ssd_on()) dma_ssd_read(c, 0, k, e[k], false);\n")
rep("p100_dma.cuh",
    "                    if (c->sc_look)\n"
    "                        std::fprintf(stderr, \"fnx ssd: device %d: the cache had %llu of %llu experts asked for (%.0f %%)\\n\", c->dev, c->sc_hit, c->sc_look,\n"
    "                                     100.0 * (double) c->sc_hit / (double) c->sc_look);\n"
    "                    c->sc_look = c->sc_hit = 0;\n",
    "                    if (c->sc_look || c->sc_kept) {   // ssd3_patch.py: the RAM tier's hits, the evicted experts it kept, what it holds\n"
    "                        int held = 0, n = 0;\n"
    "                        if (c->sc) {\n"
    "                            std::lock_guard<std::mutex> lk(c->sc->mu);\n"
    "                            for (const SsdClass& s : c->sc->cls) held += s.held, n += s.n;\n"
    "                        }\n"
    "                        std::fprintf(stderr, \"fnx ssd: device %d: the RAM tier had %llu of %llu experts asked for (%.0f %%), kept %llu evicted, holds %d of %d\\n\",\n"
    "                                     c->dev, c->sc_hit, c->sc_look, c->sc_look ? 100.0 * (double) c->sc_hit / (double) c->sc_look : 0.0, c->sc_kept, held, n);\n"
    "                    }\n"
    "                    c->sc_look = c->sc_hit = c->sc_kept = 0;\n")
rep("p100_dma.cuh",
    "    if (c->sc) {\n"
    "        cudaFreeHost(c->sc->mem);\n"
    "        delete c->sc;\n"
    "    }\n",
    "    if (c->sc) {   // ssd3_patch.py\n"
    "        for (SsdClass& s : c->sc->cls)\n"
    "            if (s.mem) cudaFreeHost(s.mem);\n"
    "        delete c->sc;\n"
    "    }\n"
    "    for (int w = 0; w < 2; ++w) {\n"
    "        if (c->csd[w]) {\n"
    "            cudaStreamSynchronize(c->csd[w]);\n"
    "            cudaStreamDestroy(c->csd[w]);\n"
    "            for (cudaEvent_t& v : c->evd[w])\n"
    "                if (v) cudaEventDestroy(v);\n"
    "        }\n"
    "        if (c->evl[w]) cudaEventDestroy(c->evl[w]);\n"
    "    }\n")
# the engine's side: warm, save, find
rep("p100_dma.cuh",
    "void p100_dma_wait(void* h, void* st) {\n",
    r'''// ssd3_patch.py: the RAM tier (sized here by need: `hottest` is every expert out of a slot, hottest first) filled with the hottest while
// each class has room past its spare, the coldest of them in the first slots and none marked used, so CLOCK (from slot 0) lets the
// coldest go first; the card idle (the load)
void p100_dma_ssd_warm(void* h, const std::vector<int>& hottest) {
    auto* c = (DmaCard*) h;
    SsdCache* sc = c && ssd_ram_gib() > 0 ? dma_ssd_cache(c, &hottest) : nullptr;
    if (!sc) return;
    const auto t0 = std::chrono::steady_clock::now();
    std::vector<std::pair<int, char*>> todo;
    {
        std::lock_guard<std::mutex> lk(sc->mu);
        std::vector<int> room(sc->cls.size()), pick;
        for (size_t k = 0; k < room.size(); ++k) room[k] = sc->cls[k].n - sc->cls[k].held - SSD_SPARE_NEED;
        for (const int g : hottest) {
            if (!sc_on(sc, g)) continue;
            int& r = room[(size_t) sc->cls_of[(size_t) (g / NEXP)]];
            if (r <= 0 || sc->slot_of[(size_t) g] >= 0) continue;
            --r;
            pick.push_back(g);
        }
        for (auto it = pick.rbegin(); it != pick.rend(); ++it)
            if (char* at = sc_take(sc, -1, *it)) {
                sc->cls[(size_t) sc->cls_of[(size_t) (*it / NEXP)]].ref[(size_t) sc->slot_of[(size_t) *it]] = 0;
                todo.push_back({*it, at});
            }
    }
    ssd_parallel(todo.size(), ssd_threads(), [&](size_t i, std::vector<char>& tmp) {
        const int g = todo[i].first;
        const SsdLayer& s = ssd_layer(c->dev, g);
        ssd_read_gu(c->dev, g, todo[i].second);
        ssd_read_dn(c->dev, g, todo[i].second + 2 * s.gu_half, tmp);
    });
    std::fprintf(stderr, "fnx ssd: device %d: the RAM tier warmed with %zu of the %zu experts out of a slot in %.1f s\n", c->dev, todo.size(),
                 hottest.size(), std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    ssd_report("the RAM tier warmed");
}
// slots' experts copied down into the RAM tier on st (those not there already), before the slots are lent; the copier idle
void p100_dma_ssd_save(void* h, const std::vector<SsdFill>& slots, void* st) {
    auto* c = (DmaCard*) h;
    SsdCache* sc = c && ssd_ram_gib() > 0 ? dma_ssd_cache(c) : nullptr;
    if (!sc) return;
    std::lock_guard<std::mutex> lk(sc->mu);
    for (const SsdFill& f : slots) {
        if (!sc_on(sc, f.g)) continue;
        if (sc->slot_of[(size_t) f.g] >= 0) {
            sc_hit(sc, -1, f.g);
            continue;
        }
        char* at = sc_take(sc, -1, f.g);
        if (!at) continue;   // no slot: not kept
        if (cudaMemcpyAsync(at, f.gu, f.gs, cudaMemcpyDeviceToHost, (cudaStream_t) st) != cudaSuccess ||
            cudaMemcpyAsync(at + f.gs, f.dn, f.ds, cudaMemcpyDeviceToHost, (cudaStream_t) st) != cudaSuccess)
            throw std::runtime_error(std::string("FNX_SSD_RAM_GIB: a slot down: ") + cudaGetErrorString(cudaGetLastError()));
    }
}
// g's bytes in the RAM tier (gate|up, then down), or null; the copier idle (a prompt pass, which stages every expert: no mark for CLOCK)
const char* p100_dma_ssd_find(void* h, int g) {
    auto* c = (DmaCard*) h;
    SsdCache* sc = c && ssd_ram_gib() > 0 ? dma_ssd_cache(c) : nullptr;
    if (!sc) return nullptr;
    std::lock_guard<std::mutex> lk(sc->mu);
    return sc->slot_of[(size_t) g] >= 0 ? sc_at(sc, g) : nullptr;
}

void p100_dma_wait(void* h, void* st) {
''')
rep("p100_over.hpp",
    "void p100_dma_wait(void* h, void* st);\n",
    "void p100_dma_wait(void* h, void* st);\n"
    "// ssd3_patch.py (FNX_SSD_EXPERTS with FNX_SSD_RAM_GIB): the copier's RAM tier (p100_dma.cuh) filled at the load with experts out of a slot,\n"
    "// hottest first; slots' experts copied down into it on st before they are lent; an expert's bytes there (gate|up, then down), or null\n"
    "void p100_dma_ssd_warm(void* h, const std::vector<int>& hottest);\n"
    "void p100_dma_ssd_save(void* h, const std::vector<SsdFill>& slots, void* st);\n"
    "const char* p100_dma_ssd_find(void* h, int g);\n")
rep("p100_over.hpp", '#include "kernels.hpp"\n', '#include "kernels.hpp"\n#include "ssd.hpp"   // ssd3_patch.py: SsdFill\n')

# ---------------------------------------------------------------- engine.cu: the warm fill, the prompt slots' experts saved
rep("engine.cu",
    "    for (CardBuf* b : cb_) { ck(cudaSetDevice(b->dev), \"set\"); ck(cudaDeviceSynchronize(), \"init sync\"); }\n",
    "    for (CardBuf* b : cb_) { ck(cudaSetDevice(b->dev), \"set\"); ck(cudaDeviceSynchronize(), \"init sync\"); }\n"
    "    if (ssd_on() && ssd_ram_gib() > 0)   // ssd3_patch.py: each card's RAM tier filled with the hottest experts out of a slot (the profile's score)\n"
    "        for (int c = 0; c < NCARD; ++c) {\n"
    "            const CardW& w = m_.cards[c];\n"
    "            if (!w.ec.n_slots || !cb_[c]->dma) continue;\n"
    "            std::vector<int> slot_of((size_t) NL * NEXP);\n"
    "            ck(cudaSetDevice(w.dev), \"set\");\n"
    "            ck(cudaMemcpy(slot_of.data(), w.ec.slot_of, slot_of.size() * sizeof(int), cudaMemcpyDeviceToHost), \"ssd warm: slots\");\n"
    "            std::vector<int> out;\n"
    "            for (int g = 0; g < NL * NEXP; ++g)\n"
    "                if (slot_of[(size_t) g] < 0 && m_.kept(g)) out.push_back(g);\n"
    "            if (m_.score.size() == (size_t) NL * NEXP)\n"
    "                std::stable_sort(out.begin(), out.end(), [&](int a, int b2) { return m_.score[(size_t) a] > m_.score[(size_t) b2]; });\n"
    "            p100_dma_ssd_warm(cb_[c]->dma, out);\n"
    "        }\n")
rep("engine.cu",
    "                if (pf_tail_[c][p] > 0) p100_tail_slots(ec, p, pf_tail_[c][p], lend, b.st);\n",
    "                if (pf_tail_[c][p] > 0) {\n"
    "                    if (lend && ssd_on() && ssd_ram_gib() > 0 && b.dma) {   // ssd3_patch.py: the slots' experts down into the RAM tier first\n"
    "                        const int cnt = pf_tail_[c][p], s1 = ec.s0[p] + ec.n[p];\n"
    "                        std::vector<int> own((size_t) cnt);\n"
    "                        ck(cudaMemcpy(own.data(), ec.owner + (s1 - cnt), (size_t) cnt * sizeof(int), cudaMemcpyDeviceToHost), \"ssd: the slots' owners\");\n"
    "                        std::vector<SsdFill> keep;\n"
    "                        for (int i = 0; i < cnt; ++i)\n"
    "                            if (const int g = own[(size_t) i]; g >= 0) {\n"
    "                                char* at = ec.pool[p] + (size_t) (s1 - cnt + i - ec.s0[p]) * ec.slot_bytes[p];\n"
    "                                keep.push_back({g, at, at + ec.dn_off[p], ec.dn_off[p], ec.slot_bytes[p] - ec.dn_off[p]});\n"
    "                            }\n"
    "                        p100_dma_ssd_save(b.dma, keep, b.st);\n"
    "                    }\n"
    "                    p100_tail_slots(ec, p, pf_tail_[c][p], lend, b.st);\n"
    "                }\n")
rep("engine.cu",
    "            ck(cudaStreamSynchronize(b.st), \"prompt slots\");\n"
    "        }\n"
    "    pf_tails_lent_ = lend;\n",
    "            if (!(lend && ssd_on() && ssd_ram_gib() > 0)) ck(cudaStreamSynchronize(b.st), \"prompt slots\");\n"
    "        }\n"
    "    if (lend && ssd_on() && ssd_ram_gib() > 0)   // ssd3_patch.py: the cards' copies down side by side, then each one waited for\n"
    "        for (int c = 0; c < NCARD; ++c) {\n"
    "            ck(cudaSetDevice(cb_[c]->dev), \"set\");\n"
    "            ck(cudaStreamSynchronize(cb_[c]->st), \"prompt slots\");\n"
    "        }\n"
    "    pf_tails_lent_ = lend;\n")

# ---------------------------------------------------------------- prefill.cu: a read stages from the RAM tier what it holds
rep("prefill.cu",
    "    std::vector<int> ssd_list[NL];\n"
    "    size_t ssd_gs[NL] = {}, ssd_ds[NL] = {};\n",
    "    std::vector<int> ssd_list[NL];\n"
    "    size_t ssd_gs[NL] = {}, ssd_ds[NL] = {};\n"
    "    std::vector<int> ssd_pos[NL];   // ssd3_patch.py: ssd_list[l][j]'s place in the stage (the RAM tier's experts take the others)\n"
    "    int ssd_fit[NL] = {};           // the layer's staged experts (the stage's down parts start after their gate|up parts)\n")
rep("prefill.cu",
    "                    p.ssd_list[l].clear();\n"
    "                    p.ssd_gs[l] = gs, p.ssd_ds[l] = ds;\n",
    "                    p.ssd_list[l].clear();\n"
    "                    p.ssd_pos[l].clear();\n"
    "                    p.ssd_gs[l] = gs, p.ssd_ds[l] = ds;\n"
    "                    p.ssd_fit[l] = fit;\n")
rep("prefill.cu",
    "                        if (ssd_on()) {\n"
    "                            p.ssd_list[l].push_back((int) i);   // ssd_patch.py: its bytes come through the ring\n"
    "                        } else {\n",
    "                        if (ssd_on()) {\n"
    "                            if (const char* hc = b.dma ? p100_dma_ssd_find(b.dma, (int) i) : nullptr) {   // ssd3_patch.py: from the RAM tier\n"
    "                                add(hc, gu_at + (size_t) k * gs, gs);\n"
    "                                add(hc + gs, dn_at + (size_t) k * ds, ds);\n"
    "                            } else {\n"
    "                                p.ssd_list[l].push_back((int) i);   // ssd_patch.py: its bytes come through the ring\n"
    "                                p.ssd_pos[l].push_back(k);\n"
    "                            }\n"
    "                        } else {\n")
rep("prefill.cu",
    "                if (ssd_on() && k) cp.push_back({nullptr, gu_at, (size_t) k * (gs + ds)});   // ssd_patch.py: the layer's one copy (src: its ring slot)\n",
    "")
rep("prefill.cu",
    "                for (const Pf::Copy& c : p.copies[l]) ckc(cudaMemcpyAsync(c.dst, p.ssd_ring[l % p.ssd_ring_n], c.n, cudaMemcpyHostToDevice, p.cs), \"stage copy\");\n",
    "                for (const Pf::Copy& c : p.copies[l]) ckc(cudaMemcpyAsync(c.dst, c.src, c.n, cudaMemcpyHostToDevice, p.cs), \"stage copy\");   // ssd3_patch.py: the RAM tier's\n"
    "                {   // the ring's, each to its place\n"
    "                    const char* r = p.ssd_ring[l % p.ssd_ring_n];\n"
    "                    const size_t gs = p.ssd_gs[l], ds = p.ssd_ds[l], n = p.ssd_list[l].size();\n"
    "                    char* gu_at = p.stage[l & 1];\n"
    "                    char* dn_at = gu_at + (size_t) p.ssd_fit[l] * gs;\n"
    "                    for (size_t j = 0; j < n; ++j) {\n"
    "                        const size_t k = (size_t) p.ssd_pos[l][j];\n"
    "                        ckc(cudaMemcpyAsync(gu_at + k * gs, r + j * gs, gs, cudaMemcpyHostToDevice, p.cs), \"stage copy\");\n"
    "                        ckc(cudaMemcpyAsync(dn_at + k * ds, r + n * gs + j * ds, ds, cudaMemcpyHostToDevice, p.cs), \"stage copy\");\n"
    "                    }\n"
    "                }\n")
print("ssd3_patch.py: the RAM tier (ExpertCache::prev, DmaBox::prev, warm, save, find) in model.hpp, model.cpp, moe.cuh, p100_moe_rows.cuh, "
      "p100_dma_box.cuh, p100_dma.cuh, p100_over.hpp, engine.cu, prefill.cu")
