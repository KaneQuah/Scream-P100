#!/usr/bin/env python3
# prompt_peer_patch.py: FNX_PF_PEER=1 on NVIDIA, the prompt path's exchange over peer access. The chunk sums of the prompt path's
# ring (pf_ring_run) and the attention partials (FNX_PF_APEER, on with it) go card to card into the receivers' DEVICE buffers, not
# through page-locked host memory; the flags stay in host memory, set by the sender's stream after its copy. On a machine whose
# cards share one x4 uplink the host exchange is what limits reading (one 16,384-row pass moved ~128 GB through that uplink).
# Needs peer_patch.py's probe to pass (FNX_PEER=1; FNX_PEER_IOMMU_TRUST=1 where the IOMMU is translating and the probe still passes),
# FNX_PF_DMA=1 and FNX_PFX_ATTN_DMA=1 (both default). Off unless FNX_PF_PEER=1. Run after peer_patch.py, in the engine's source folder.
import sys
from pathlib import Path

S = Path("src")
if not (S / "engine.cu").exists():
    sys.exit("prompt_peer_patch.py: run it in the folder that holds src/")
if "prompt_peer" in (S / "prefill.cu").read_text(encoding="utf-8"):
    sys.exit("prompt_peer_patch.py: applied already")


def rep(name, old, new):
    p = S / name
    t = p.read_text(encoding="utf-8")
    n = t.count(old)
    if n != 1:
        sys.exit(f"prompt_peer_patch.py: {name}: {n} matches, want 1:\n{old[:300]}")
    p.write_text(t.replace(old, new), encoding="utf-8")


rep('engine.cu',
    '        else std::fprintf(stderr, "fnx: FNX_PEER=0: the host exchange, no peer probe\\n");\n        peer_on = pd.use;\n        peer_parts = peer_on ? (int) env_int("FNX_PEER_PARTS", 3) : 0;\n        if (peer_on && (peer_parts < 1 || peer_parts > 3)) throw std::runtime_error("FNX_PEER_PARTS: 1, 2 or 3");\n',
    '        else std::fprintf(stderr, "fnx: FNX_PEER=0: the host exchange, no peer probe\\n");\n        peer_on = pd.use;\n        peer_on_ = pd.use;\n        peer_parts = peer_on ? (int) env_int("FNX_PEER_PARTS", 3) : 0;\n        if (peer_on && (peer_parts < 1 || peer_parts > 3)) throw std::runtime_error("FNX_PEER_PARTS: 1, 2 or 3");\n')

rep('engine.hpp',
    '\nclass Engine {\npublic:\n    // lanes: how many add_lane() blocks the K/V pool has room for on top of max_ctx positions\n',
    '\nclass Engine {\n    bool peer_on_ = false;   // prompt_peer: peer access between every pair is on (the FNX_PEER probe passed)\npublic:\n    // lanes: how many add_lane() blocks the K/V pool has room for on top of max_ctx positions\n')

rep('pf_common.cuh',
    '                // the slot that arrived in host memory goes to device memory on the copy engine (asynchronous, beside the next chunk\'s\n                // compute), and the add reads it there: a kernel reading host memory stalls its SMs at PCIe speed (FNX_PF_DMA=0: it reads the host slot)\n                if (rows(o.part) > 0 && r.dma) {\n                    ckc(cudaMemcpyAsync(r.din[o.hop][set], r.in[o.hop][set], (size_t) rows(o.part) * N * sizeof(__half), cudaMemcpyDefault, s),\n                        "ring in");\n',
    '                // the slot that arrived in host memory goes to device memory on the copy engine (asynchronous, beside the next chunk\'s\n                // compute), and the add reads it there: a kernel reading host memory stalls its SMs at PCIe speed (FNX_PF_DMA=0: it reads the host slot)\n                if (rows(o.part) > 0 && r.dma && !r.peer) {   // (r.peer: the sender wrote din directly)\n                    ckc(cudaMemcpyAsync(r.din[o.hop][set], r.in[o.hop][set], (size_t) rows(o.part) * N * sizeof(__half), cudaMemcpyDefault, s),\n                        "ring in");\n')

rep('pf_common.cuh',
    '                        ckc(cudaMemcpyAsync(r.nag[set] + off, r.dag[set] + off, (size_t) rows(o.part) * N * sizeof(__half), cudaMemcpyDefault, s),\n                            "ring gather");\n                    else   // a part that arrived (three or more cards), or no DMA: host -> host by a kernel (a host-to-host memcpy would block the thread)\n                        k_pfr_copy<<<grid_for((size_t) rows(o.part) * N / 8), 256, 0, s>>>((const uint4*) (r.ag[set] + off), (uint4*) (r.nag[set] + off),\n',
    '                        ckc(cudaMemcpyAsync(r.nag[set] + off, r.dag[set] + off, (size_t) rows(o.part) * N * sizeof(__half), cudaMemcpyDefault, s),\n                            "ring gather");\n                    else if (r.peer)   // a part that arrived (three or more cards): this card\'s device rows -> the next card\'s device rows\n                        ckc(cudaMemcpyAsync(r.nag[set] + off, r.dag[set] + off, (size_t) rows(o.part) * N * sizeof(__half), cudaMemcpyDefault, s),\n                            "ring gather peer");\n                    else   // a part that arrived (three or more cards), or no DMA: host -> host by a kernel (a host-to-host memcpy would block the thread)\n                        k_pfr_copy<<<grid_for((size_t) rows(o.part) * N / 8), 256, 0, s>>>((const uint4*) (r.ag[set] + off), (uint4*) (r.nag[set] + off),\n')

rep('pf_common.cuh',
    '                    // the received slot to device memory by DMA (as ADD), the sum stored into this card\'s device rows and every other card\'s\n                    // host rows (stores to host memory are posted, they do not stall the SMs as loads do)\n                    if (r.dma) {\n                        ckc(cudaMemcpyAsync(r.din[o.hop][set], r.in[o.hop][set], (size_t) rows(o.part) * N * sizeof(__half), cudaMemcpyDefault, s),\n                            "ring in");\n',
    '                    // the received slot to device memory by DMA (as ADD), the sum stored into this card\'s device rows and every other card\'s\n                    // host rows (stores to host memory are posted, they do not stall the SMs as loads do)\n                    if (r.dma && !r.peer) {\n                        ckc(cudaMemcpyAsync(r.din[o.hop][set], r.in[o.hop][set], (size_t) rows(o.part) * N * sizeof(__half), cudaMemcpyDefault, s),\n                            "ring in");\n')

rep('pf_common.cuh',
    '                }\n                // the parts that arrived (everything but the one finished here) go to device memory by DMA, then one kernel reads it all\n                if (r.dma)\n                    for (int p = 0; p < NCARD; ++p)\n                        if (p != fin_part && rows(p) > 0)\n',
    '                }\n                // the parts that arrived (everything but the one finished here) go to device memory by DMA, then one kernel reads it all\n                if (r.dma && !r.peer)\n                    for (int p = 0; p < NCARD; ++p)\n                        if (p != fin_part && rows(p) > 0)\n')

rep('prefill.cu',
    '    bool dma_check = false;  // FNX_PF_DMA_CHECK=1: compare each DMA copy with its host source by a kernel\n    bool dma = true;         // receive and gather by DMA into device memory (FNX_PF_DMA=0: kernels read host memory)\n    std::vector<PfRingOp> plan;\n};\n',
    "    bool dma_check = false;  // FNX_PF_DMA_CHECK=1: compare each DMA copy with its host source by a kernel\n    bool dma = true;         // receive and gather by DMA into device memory (FNX_PF_DMA=0: kernels read host memory)\n    bool peer = false;       // prompt_peer (FNX_PF_PEER=1): nin / nag / pag point into the next cards' DEVICE buffers (din / dag): no host rows, no receive DMA\n    std::vector<PfRingOp> plan;\n};\n")

rep('prefill.cu',
    "    float* parecv[2][NCARD];\n    unsigned* paflags[2][NCARD];\n    // FNX_PFX_ATTN_DMA (default on): the other cards' partials are copied from the host receive buffer into device memory by the copy engine on st3, piece by\n    // piece as their flags arrive, so the merge reads device memory (reading the mapped host buffer from the merge kernel stalls its SMs at PCIe speed)\n",
    "    float* parecv[2][NCARD];\n    unsigned* paflags[2][NCARD];\n    // FNX_PF_PEER with FNX_PF_APEER (default 1): the partials go card to card into the receivers' arecv_dev (no host buffer, no receive copy); flags stay in host memory\n    bool apeer = false;\n    float* parecv_dev[2][NCARD] = {};\n    // FNX_PFX_ATTN_DMA (default on): the other cards' partials are copied from the host receive buffer into device memory by the copy engine on st3, piece by\n    // piece as their flags arrive, so the merge reads device memory (reading the mapped host buffer from the merge kernel stalls its SMs at PCIe speed)\n")

rep('prefill.cu',
    '    for (int d = 1; d < NCARD; ++d) {\n        const int k = (g + d) % NCARD, slot = (g - k + NCARD) % NCARD - 1;\n        ckc(cudaMemcpyAsync(p.parecv[set][k] + slot * blk + off, p.apart[set] + k * blk + off,\n                            attn_prompt_block_floats(nr) * sizeof(float), cudaMemcpyDefault, s), "attn copy");   // device -> host buffer\n        cud(cuStreamWriteValue32((CUstream) s, (CUdeviceptr) (p.paflags[set][k] + slot * ATT_PIECES_MAX + i), seq, 0), "attn flag");\n    }\n',
    '    for (int d = 1; d < NCARD; ++d) {\n        const int k = (g + d) % NCARD, slot = (g - k + NCARD) % NCARD - 1;\n        float* dst = p.apeer ? p.parecv_dev[set][k] : p.parecv[set][k];\n        ckc(cudaMemcpyAsync(dst + slot * blk + off, p.apart[set] + k * blk + off,\n                            attn_prompt_block_floats(nr) * sizeof(float), cudaMemcpyDefault, s), "attn copy");   // device -> host buffer (or the peer\'s device buffer)\n        cud(cuStreamWriteValue32((CUstream) s, (CUdeviceptr) (p.paflags[set][k] + slot * ATT_PIECES_MAX + i), seq, 0), "attn flag");\n    }\n')

rep('prefill.cu',
    '            for (int slot = 0; slot < NCARD - 1; ++slot) {\n                cud(cuStreamWaitValue32((CUstream) p.st3, (CUdeviceptr) (p.aflags[set] + slot * ATT_PIECES_MAX + i), seq, CU_STREAM_WAIT_VALUE_GEQ), "attn wait (dma)");\n                ckc(cudaMemcpyAsync(p.arecv_dev[set] + slot * blk + off, p.arecv[set] + slot * blk + off, attn_prompt_block_floats(r1 - r0) * sizeof(float),\n                                    cudaMemcpyDefault, p.st3), "attn recv copy");   // host buffer -> device, on the copy engine\n',
    '            for (int slot = 0; slot < NCARD - 1; ++slot) {\n                cud(cuStreamWaitValue32((CUstream) p.st3, (CUdeviceptr) (p.aflags[set] + slot * ATT_PIECES_MAX + i), seq, CU_STREAM_WAIT_VALUE_GEQ), "attn wait (dma)");\n                if (!p.apeer)\n                ckc(cudaMemcpyAsync(p.arecv_dev[set] + slot * blk + off, p.arecv[set] + slot * blk + off, attn_prompt_block_floats(r1 - r0) * sizeof(float),\n                                    cudaMemcpyDefault, p.st3), "attn recv copy");   // host buffer -> device, on the copy engine\n')

rep('prefill.cu',
    "            pf_[c]->pdev[k] = pf_[k]->dev;\n        }\n    if (g_lat_split)\n        for (int c = 0; c < NCARD; ++c) {   // edge0_patch6.py: each card's latent halves land in the other card's host rows\n",
    '            pf_[c]->pdev[k] = pf_[k]->dev;\n        }\n    if (env_int("FNX_PF_PEER", 0) != 0 && env_int("FNX_PF_APEER", 1) != 0) {\n        if (!peer_on_ || !pf_[0]->attn_dma) throw std::runtime_error("FNX_PF_APEER needs peer access and FNX_PFX_ATTN_DMA=1");\n        for (int c = 0; c < NCARD; ++c) {\n            pf_[c]->apeer = true;\n            for (int s = 0; s < 2; ++s)\n                for (int k = 0; k < NCARD; ++k) pf_[c]->parecv_dev[s][k] = pf_[k]->arecv_dev[s];\n        }\n        std::fprintf(stderr, "fnx: FNX_PF_APEER: the attention partials go card to card (device buffers)\\n");\n    }\n    if (g_lat_split)\n        for (int c = 0; c < NCARD; ++c) {   // edge0_patch6.py: each card\'s latent halves land in the other card\'s host rows\n')

rep('prefill.cu',
    '            r.pflags[d] = pf_[d]->ring_ce.flags;\n            for (int k = 0; k < 2; ++k) r.pag[d][k] = pf_[d]->ring_ce.ag[k];\n        }\n        // the all-gather: HIP defaults to the direct gather (a ring of host-to-host forwards costs 1/3 of the prompt time on four cards),\n',
    '            r.pflags[d] = pf_[d]->ring_ce.flags;\n            for (int k = 0; k < 2; ++k) r.pag[d][k] = pf_[d]->ring_ce.ag[k];\n        }\n        // prompt_peer: FNX_PF_PEER=1 with peer access on between every pair: the sums go card to card into the receivers\' device buffers\n        // (din, dag), the flags stay in host memory (set by the sender\'s stream after its copy). Needs the DMA layout (device din / dag)\n        if (env_int("FNX_PF_PEER", 0) != 0) {\n            if (!peer_on_) throw std::runtime_error("FNX_PF_PEER=1 needs peer access (FNX_PEER=1 and the start probe passing)");\n            if (!env_int("FNX_PF_DMA", 1)) throw std::runtime_error("FNX_PF_PEER=1 needs FNX_PF_DMA=1");\n            p.ring_ce.peer = true;\n            for (int k = 0; k < 2; ++k) {\n                for (int h = 0; h < NCARD - 1; ++h) p.ring_ce.nin[h][k] = n.ring_ce.din[h][k];\n                p.ring_ce.nag[k] = n.ring_ce.dag[k];\n                for (int d = 0; d < NCARD; ++d) p.ring_ce.pag[d][k] = pf_[d]->ring_ce.dag[k];\n            }\n            if (c == 0) std::fprintf(stderr, "fnx: FNX_PF_PEER: the prompt path\'s chunk sums go card to card (device buffers), flags in host memory\\n");\n        }\n        // the all-gather: HIP defaults to the direct gather (a ring of host-to-host forwards costs 1/3 of the prompt time on four cards),\n')
