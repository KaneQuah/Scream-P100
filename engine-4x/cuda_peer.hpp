// UNVERIFIED PROOF OF CONCEPT: nobody has run this on four cards, or on any NVIDIA cards with peer access (engine-4x/README.md).
//
// engine-4x (peer_patch.py copies this file to src/): peer access between NVIDIA cards, and the start probe that decides whether
// the window sums and the attention merge are pushed into the other cards' memory (FNX_PEER; the rules are hip/peer_policy.hpp's,
// the same as on AMD) or go through page-locked host memory as in every other build. Ported from hip/peer.h, with three differences:
//   - every card's IOMMU group is read first. A translated domain (type DMA or DMA-FQ) faults every peer write (dmesg: "DMAR: ...
//     PTE Write access is not set"), so there is no probe then, only the host exchange and a line that says why
//   - nothing reads a peer's memory, the probe included: a P2P read that the platform does not route can end the CUDA context, and
//     the engine only pushes (the sender stores into the receiver's memory, the receiver reads its own)
//   - the round trips are bounded in clock64 cycles (sm_60 has no wall clock register)
// Whatever fails, peer access is switched off again on every pair before the engine goes on with the host exchange.
#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "card_check.hpp"
#include "hip/peer_policy.hpp"

namespace fnx {

namespace cuda_peer_detail {
__global__ void k_fill(unsigned* p, size_t n, unsigned seed) {   // the sender's stores into the receiver's buffer
    for (size_t e = blockIdx.x * (size_t) blockDim.x + threadIdx.x; e < n; e += (size_t) gridDim.x * blockDim.x) p[e] = seed ^ (unsigned) (e * 2654435761u);
}
__global__ void k_cmp(const unsigned* p, size_t n, unsigned seed, unsigned* bad) {   // the receiver's reads of its own buffer
    unsigned b = 0;
    for (size_t e = blockIdx.x * (size_t) blockDim.x + threadIdx.x; e < n; e += (size_t) gridDim.x * blockDim.x) b += p[e] != (seed ^ (unsigned) (e * 2654435761u));
    if (b) atomicAdd(bad, b);
}
// flag ping-pong: each side stores into the other's flag and waits on its own; `trips` round trips, each wait bounded (about 3 s at
// the P100's clocks); ok[0] = 1 if this side completed
__global__ void k_ping(volatile unsigned* mine, volatile unsigned* theirs, int trips, int first, unsigned* ok) {
    if (threadIdx.x != 0) return;
    const long long limit = 4000000000LL;
    for (int i = 1; i <= trips; ++i) {
        if (first) { *theirs = (unsigned) i; __threadfence_system(); }
        const long long t0 = clock64();
        while (*mine < (unsigned) i)
            if (clock64() - t0 > limit) return;
        if (!first) { *theirs = (unsigned) i; __threadfence_system(); }
    }
    *ok = 1;
}
}  // namespace cuda_peer_detail

// a card's IOMMU domain type from sysfs: "" when the card has no iommu_group (no IOMMU in the way), "?" when it cannot be read
inline std::string card_iommu_type(int dev) {
    char bus[32];
    if (cudaDeviceGetPCIBusId(bus, sizeof bus, dev) != cudaSuccess) { cudaGetLastError(); return "?"; }
    std::string bdf;
    try { bdf = sysfs_bdf(bus); } catch (const std::exception&) { return "?"; }
    const std::string dir = "/sys/bus/pci/devices/" + bdf;
    std::ifstream probe(dir + "/vendor");
    if (!probe) return "?";
    std::ifstream link(dir + "/iommu_group/type");
    if (!link) {
        std::ifstream grp(dir + "/iommu_group/reserved_regions");   // the group exists but its type cannot be read
        return grp ? "?" : "";
    }
    std::string t;
    std::getline(link, t);
    while (!t.empty() && (t.back() == '\n' || t.back() == ' ')) t.pop_back();
    return t;
}

inline void cuda_peer_disable_all(const std::vector<int>& devs) {
    for (int a : devs) {
        if (cudaSetDevice(a) != cudaSuccess) { cudaGetLastError(); continue; }
        for (int b : devs)
            if (a != b && cudaDeviceDisablePeerAccess(b) != cudaSuccess) cudaGetLastError();   // not enabled: nothing to undo
    }
}

#define FNX_CPEER_CK(x, what)                                       \
    do {                                                            \
        const cudaError_t e_ = (x);                                 \
        if (e_ != cudaSuccess) {                                    \
            cudaGetLastError();                                     \
            why = std::string(what) + ": " + cudaGetErrorString(e_); \
            cuda_peer_disable_all(devs);                            \
            return false;                                           \
        }                                                           \
    } while (0)

// The IOMMU groups, then CUDA's own answer for every pair, then access enabled between every pair, then every directed pair probed:
// 4 MiB stored by the sender into the receiver's memory and compared in full by the receiver (its own reads), the stores timed,
// and 200 flag round trips with each side storing into the other's memory. why: what passed (rates and the slowest round trip), or
// what failed; on failure peer access is off again on every pair
inline bool cuda_peer_enable_and_probe(const std::vector<int>& devs, std::string& why) {
    using namespace cuda_peer_detail;
    for (int d : devs) {
        const std::string t = card_iommu_type(d);
        if (t == "?") { why = "card " + std::to_string(d) + ": its IOMMU group cannot be read from sysfs"; return false; }
        if (!t.empty() && t != "identity") {
            why = "card " + std::to_string(d) + " is in an IOMMU group of type '" + t +
                  "', which faults every write from another card (boot with iommu=pt, or the IOMMU off)";
            return false;
        }
    }
    for (int a : devs)
        for (int b : devs) {
            if (a == b) continue;
            int can = 0;
            if (cudaDeviceCanAccessPeer(&can, a, b) != cudaSuccess || !can) {
                cudaGetLastError();
                why = "CUDA says card " + std::to_string(a) + " cannot access card " + std::to_string(b) +
                      " (nvidia-smi topo -p2p w; cards on different CPUs, behind a chipset or an ACS port usually cannot)";
                return false;
            }
        }
    for (int a : devs) {
        FNX_CPEER_CK(cudaSetDevice(a), "set device");
        for (int b : devs) {
            if (a == b) continue;
            const cudaError_t e = cudaDeviceEnablePeerAccess(b, 0);
            if (e != cudaSuccess && e != cudaErrorPeerAccessAlreadyEnabled) {
                cudaGetLastError();
                why = "enabling access from card " + std::to_string(a) + " to card " + std::to_string(b) + ": " + cudaGetErrorString(e);
                cuda_peer_disable_all(devs);
                return false;
            }
            cudaGetLastError();   // AlreadyEnabled is sticky until read
        }
    }
    const size_t n = (size_t) 1 << 20;   // words: 4 MiB
    double worst_rt = 0.0, lo_gbs = 1e300, hi_gbs = 0.0;
    for (size_t ia = 0; ia < devs.size(); ++ia)
        for (size_t ib = 0; ib < devs.size(); ++ib) {
            if (ia == ib) continue;
            const int a = devs[ia], b = devs[ib];
            const std::string pair = "card " + std::to_string(a) + " -> card " + std::to_string(b);
            unsigned *buf = nullptr, *bad_b = nullptr, *fa = nullptr, *fb = nullptr, *oka = nullptr, *okb = nullptr;
            cudaEvent_t e0 = nullptr, e1 = nullptr;
            cudaStream_t sa = nullptr, sb = nullptr;
            const unsigned seed = 0x5eed0000u + (unsigned) (a * 16 + b);
            bool pass = true;
            std::string pwhy;
            auto fail = [&](const std::string& s) { pass = false; pwhy = pair + ": " + s; cudaGetLastError(); };
            do {
                unsigned h = 0;
                if (cudaSetDevice(b) != cudaSuccess || cudaMalloc((void**) &buf, n * 4) != cudaSuccess || cudaMemset(buf, 0, n * 4) != cudaSuccess ||
                    cudaMalloc((void**) &bad_b, 4) != cudaSuccess || cudaMemset(bad_b, 0, 4) != cudaSuccess ||
                    cudaMalloc((void**) &fb, 256) != cudaSuccess || cudaMemset(fb, 0, 256) != cudaSuccess ||
                    cudaMalloc((void**) &okb, 4) != cudaSuccess || cudaMemset(okb, 0, 4) != cudaSuccess ||
                    cudaStreamCreateWithFlags(&sb, cudaStreamNonBlocking) != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess) {
                    fail("allocation on the receiving card");
                    break;
                }
                if (cudaSetDevice(a) != cudaSuccess || cudaMalloc((void**) &fa, 256) != cudaSuccess || cudaMemset(fa, 0, 256) != cudaSuccess ||
                    cudaMalloc((void**) &oka, 4) != cudaSuccess || cudaMemset(oka, 0, 4) != cudaSuccess ||
                    cudaStreamCreateWithFlags(&sa, cudaStreamNonBlocking) != cudaSuccess || cudaEventCreate(&e0) != cudaSuccess ||
                    cudaEventCreate(&e1) != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess) {
                    fail("allocation on the sending card");
                    break;
                }
                // 1. the sender stores into the receiver's memory: once to warm the path, then timed
                k_fill<<<64, 256, 0, sa>>>(buf, n, seed ^ 0xffffffffu);
                cudaEventRecord(e0, sa);
                k_fill<<<64, 256, 0, sa>>>(buf, n, seed);
                cudaEventRecord(e1, sa);
                if (cudaStreamSynchronize(sa) != cudaSuccess) { fail("the store kernel failed"); break; }
                float ms = 0.f;
                cudaEventElapsedTime(&ms, e0, e1);
                // 2. the receiver compares in full (its own reads)
                if (cudaSetDevice(b) != cudaSuccess) { fail("set device"); break; }
                k_cmp<<<64, 256>>>(buf, n, seed, bad_b);
                if (cudaDeviceSynchronize() != cudaSuccess || cudaMemcpy(&h, bad_b, 4, cudaMemcpyDeviceToHost) != cudaSuccess) { fail("the compare kernel failed"); break; }
                if (h) { fail(std::to_string(h) + " of " + std::to_string(n) + " words arrived wrong"); break; }
                if (ms > 0.f) {
                    const double gbs = n * 4.0 / (ms * 1e6);
                    lo_gbs = std::min(lo_gbs, gbs);
                    hi_gbs = std::max(hi_gbs, gbs);
                }
                // 3. flag round trips: a first, b second, each side storing into the other's flag, both kernels bounded
                const int trips = 200;
                const auto t0 = std::chrono::steady_clock::now();
                cudaSetDevice(b); k_ping<<<1, 32, 0, sb>>>(fb, fa, trips, 0, okb);
                cudaSetDevice(a); k_ping<<<1, 32, 0, sa>>>(fa, fb, trips, 1, oka);
                cudaSetDevice(a); const cudaError_t ea = cudaStreamSynchronize(sa);
                cudaSetDevice(b); const cudaError_t eb = cudaStreamSynchronize(sb);
                const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / trips;
                unsigned va = 0, vb = 0;
                cudaSetDevice(a); cudaMemcpy(&va, oka, 4, cudaMemcpyDeviceToHost);
                cudaSetDevice(b); cudaMemcpy(&vb, okb, 4, cudaMemcpyDeviceToHost);
                if (ea != cudaSuccess || eb != cudaSuccess || !va || !vb) { fail("the flag round trip over peer memory did not complete"); break; }
                worst_rt = std::max(worst_rt, us);
            } while (false);
            cudaSetDevice(b);
            if (buf) cudaFree(buf);
            if (bad_b) cudaFree(bad_b);
            if (fb) cudaFree(fb);
            if (okb) cudaFree(okb);
            if (sb) cudaStreamDestroy(sb);
            cudaSetDevice(a);
            if (fa) cudaFree(fa);
            if (oka) cudaFree(oka);
            if (sa) cudaStreamDestroy(sa);
            if (e0) cudaEventDestroy(e0);
            if (e1) cudaEventDestroy(e1);
            cudaGetLastError();
            if (!pass) {
                why = pwhy;
                cuda_peer_disable_all(devs);
                return false;
            }
        }
    char line[256];
    std::snprintf(line, sizeof line,
                  "all %zu directed pairs: 4 MiB stored and compared in full on the receiving card, %.1f to %.1f GB/s, flag round trip at most %.1f us",
                  devs.size() * (devs.size() - 1), lo_gbs, hi_gbs, worst_rt);
    why = line;
    return true;
}
#undef FNX_CPEER_CK

}  // namespace fnx
