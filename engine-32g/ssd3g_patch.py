#!/usr/bin/env python3
# ssd3g_patch.py (2026-11-19, mikro/q4h32/; after ssd_patch.py, ssd3e_patch.py and ssd3f_patch.py): a list's down copies in two parts. Until
# now each evicted expert went down whole (gate|up, then down: 1.46 MiB, ~123 us) before the gate|up part of the expert that takes its
# slot (0.97 MiB, ~82 us) came up, so the copies up of a list's gate|up parts went at the pace of whole copies down (round q4h32k a: a
# decode round 60 ms against production's ~38). Now every gate|up part goes down first, an event each, then every down part, an event each;
# a gate|up part up waits for its slot's gate|up part down only, a down part up for its slot's down part down.
import sys
from pathlib import Path

S = Path("src")
if "ssd3f_patch.py" not in (S / "engine.cu").read_text(encoding="utf-8"):
    sys.exit("ssd3g_patch.py: ssd_patch.py, ssd3e_patch.py and ssd3f_patch.py first")
if "ssd3g_patch.py" in (S / "p100_dma.cuh").read_text(encoding="utf-8"):
    sys.exit("ssd3g_patch.py: applied already")


def rep(name, old, new, count=1):
    p = S / name
    t = p.read_text(encoding="utf-8")
    n = t.count(old)
    if n != count:
        sys.exit(f"ssd3g_patch.py: {name}: {n} matches, want {count}:\n{old[:300]}")
    p.write_text(t.replace(old, new), encoding="utf-8")


rep("p100_dma.cuh",
    "    cudaEvent_t evd[2][DMA_MAX] = {};\n",
    "    cudaEvent_t evd[2][DMA_MAX] = {};\n"
    "    cudaEvent_t evn[2][DMA_MAX] = {};    // ssd3g_patch.py: evd: an entry's gate|up part down, evn: its down part down\n")
rep("p100_dma.cuh",
    "// list w's evicted experts (c->sv) copied down on the down stream w, an event each: that direction's copy engine takes them while the\n"
    "// other copies the list's experts up, each up copy waiting for its own slot's down copy only\n",
    "// list w's evicted experts (c->sv) copied down on the down stream w: that direction's copy engine takes them while the other copies the\n"
    "// list's experts up. ssd3g_patch.py: every gate|up part first (evd), then every down part (evn), so that a gate|up part up waits for its\n"
    "// slot's gate|up part down only, not for whole experts\n")
rep("p100_dma.cuh",
    "        for (int k = 0; k < DMA_MAX && ok; ++k) ok = cudaEventCreateWithFlags(&c->evd[w][k], cudaEventDisableTiming) == cudaSuccess;\n",
    "        for (int k = 0; k < DMA_MAX && ok; ++k)\n"
    "            ok = cudaEventCreateWithFlags(&c->evd[w][k], cudaEventDisableTiming) == cudaSuccess &&\n"
    "                 cudaEventCreateWithFlags(&c->evn[w][k], cudaEventDisableTiming) == cudaSuccess;\n")
rep("p100_dma.cuh",
    "    for (int k = 0; k < n; ++k)\n"
    "        if (char* d = c->sv[w][k]) {\n"
    "            cudaMemcpyAsync(d, e[k].dst_gu, gb, cudaMemcpyDeviceToHost, c->csd[w]);\n"
    "            cudaMemcpyAsync(d + gb, e[k].dst_dn, db, cudaMemcpyDeviceToHost, c->csd[w]);\n"
    "            cudaEventRecord(c->evd[w][k], c->csd[w]);\n"
    "        }\n"
    "}\n",
    "    for (int k = 0; k < n; ++k)\n"
    "        if (char* d = c->sv[w][k]) {\n"
    "            cudaMemcpyAsync(d, e[k].dst_gu, gb, cudaMemcpyDeviceToHost, c->csd[w]);\n"
    "            cudaEventRecord(c->evd[w][k], c->csd[w]);\n"
    "        }\n"
    "    for (int k = 0; k < n; ++k)\n"
    "        if (char* d = c->sv[w][k]) {\n"
    "            cudaMemcpyAsync(d + gb, e[k].dst_dn, db, cudaMemcpyDeviceToHost, c->csd[w]);\n"
    "            cudaEventRecord(c->evn[w][k], c->csd[w]);\n"
    "        }\n"
    "}\n")
# the down parts up: each once its slot's down part is down
rep("p100_dma.cuh",
    "                    if (ssd_on()) dma_ssd_read(c, 1, k, e2[k], true);   // ssd_patch.py\n",
    "                    if (ssd_on() && c->sv[1][k]) cudaStreamWaitEvent(c->cs2, c->evn[1][k], 0);   // ssd3g_patch.py: once its slot's down part is down\n"
    "                    if (ssd_on()) dma_ssd_read(c, 1, k, e2[k], true);   // ssd_patch.py\n")
rep("p100_dma.cuh",
    "            if (ssd_on()) dma_ssd_read(c, 0, k, e[k], true);   // ssd_patch.py\n",
    "            if (ssd_on() && c->sv[0][k]) cudaStreamWaitEvent(c->cs, c->evn[0][k], 0);   // ssd3g_patch.py: once its slot's down part is down\n"
    "            if (ssd_on()) dma_ssd_read(c, 0, k, e[k], true);   // ssd_patch.py\n")
rep("p100_dma.cuh",
    "            for (cudaEvent_t& v : c->evd[w])\n"
    "                if (v) cudaEventDestroy(v);\n",
    "            for (cudaEvent_t& v : c->evd[w])\n"
    "                if (v) cudaEventDestroy(v);\n"
    "            for (cudaEvent_t& v : c->evn[w])   // ssd3g_patch.py\n"
    "                if (v) cudaEventDestroy(v);\n")
print("ssd3g_patch.py: a list's down copies in two parts, gate|up first (p100_dma.cuh)")
