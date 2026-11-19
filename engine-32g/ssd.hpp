// ssd.hpp (ssd_patch.py, 2026-11-19; the user: "verzijo q4 za uporabnike ki imajo samo 32gb rama ... prefill iz ssd v ram in
// iz rama v vram ... vnaprej nalagal vsaj 2 lejerja"): FNX_SSD_EXPERTS=1 keeps no routed expert in page-locked RAM. An expert's
// host copy is its bytes in the GGUF file, read through the page cache when a card needs it, so the page cache (which the kernel
// gives back under pressure) is the RAM tier between the disk and the slots:
//   a decode window's misses: the card's copier thread (p100_dma.cuh) reads them into a page-locked bounce buffer, then copies
//   them on as before; FNX_PREFETCH's guessed experts likewise, a layer ahead
//   a prompt pass: worker threads read each layer's experts FNX_SSD_AHEAD layers ahead (default 2) into a ring of page-locked
//   buffers laid out as the layer's stage, copied in with one copy (prefill.cu); every read stages, none reads in place
//   the slots filled at the load, by the arena and by its refills: ssd_fill below
// Off (unset or 0): nothing changes. On, a host copy's pointer in ExpertCache::host_gu / host_dn and CardW::ec_host_gu /
// ec_host_dn is a tag, odd and never read as memory: (g << 2) | 1 for gate|up, | 3 for down. The slots get the same bytes as
// from page-locked copies, so the logits are those of the same settings without it.
#pragma once

#include "model.hpp"

#include <cuda_runtime.h>
#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace fnx {

inline bool ssd_on() {
    static const bool on = [] {
        const char* e = std::getenv("FNX_SSD_EXPERTS");
        return e && std::atoi(e) != 0;
    }();
    return on;
}
// FNX_SSD_THREADS: the readers of a batch (a prompt layer's experts, a fill), default 8
inline int ssd_threads() {
    static const int n = [] {
        const char* e = std::getenv("FNX_SSD_THREADS");
        return std::max(1, std::min(64, e ? std::atoi(e) : 8));
    }();
    return n;
}
// FNX_SSD_RAM_GIB: the copier's page-locked cache of the experts it read, GiB a card (default 0: none, a bounce buffer only;
// p100_dma.cuh). A hit there is copied on without a read
inline double ssd_ram_gib() {
    static const double g = [] {
        const char* e = std::getenv("FNX_SSD_RAM_GIB");
        return e ? std::max(0.0, std::atof(e)) : 0.0;
    }();
    return g;
}
// FNX_SSD_AHEAD: how many layers ahead of the one being copied in a prompt pass reads (default 2; 1 to 6)
inline int ssd_ahead() {
    static const int n = [] {
        const char* e = std::getenv("FNX_SSD_AHEAD");
        return std::max(1, std::min(6, e ? std::atoi(e) : 2));
    }();
    return n;
}

// where one layer's routed experts are in the file, and the card's part of each: gate rows from gu_skip (gu_half bytes), then
// the same rows of up; of down every row's bytes [dn_skip, dn_skip + dn_take). Experts follow each other at gu_mat / dn_mat
struct SsdLayer {
    int fd_gate = -1, fd_up = -1, fd_down = -1;
    uint64_t gate0 = 0, up0 = 0, down0 = 0;
    size_t gu_mat = 0, gu_skip = 0, gu_half = 0;
    size_t dn_mat = 0, dn_row = 0, dn_skip = 0, dn_take = 0;
    int dn_rows = 0;
};
constexpr int SSD_DEVS = 16;
inline std::vector<SsdLayer> g_ssd_layers[SSD_DEVS];   // by device, [NL]: each card's loader fills its own
inline std::vector<int> g_ssd_fd;                      // by shard, opened before the load: plain reads, the page cache's
inline std::atomic<unsigned long long> g_ssd_reads{0}, g_ssd_bytes{0}, g_ssd_ns{0};   // experts' parts read, bytes, time

inline const char* ssd_tag(int g, bool dn) { return reinterpret_cast<const char*>(((uintptr_t) g << 2) | (dn ? 3u : 1u)); }
inline bool ssd_tagged(const void* p) { return (reinterpret_cast<uintptr_t>(p) & 1u) != 0; }
inline int ssd_expert(const void* p) { return (int) (reinterpret_cast<uintptr_t>(p) >> 2); }

// the shards opened for the reads (Model's constructor, before the cards load). Random access: no read-ahead past what is asked
inline void ssd_open(const Gguf& g) {
    if (!g_ssd_fd.empty()) return;
    for (int s = 0; s < g.shards(); ++s) {
        const int fd = ::open(g.shard_path(s).c_str(), O_RDONLY);
        if (fd < 0) throw std::runtime_error("FNX_SSD_EXPERTS: cannot open " + g.shard_path(s) + ": " + std::strerror(errno));
        posix_fadvise(fd, 0, 0, POSIX_FADV_RANDOM);
        g_ssd_fd.push_back(fd);
    }
}
inline int ssd_fd(const Tensor& t) {
    if (t.shard < 0 || t.shard >= (int) g_ssd_fd.size())
        throw std::runtime_error("FNX_SSD_EXPERTS: " + t.name + " is not read from its file as it is there (converted at load)");
    return g_ssd_fd[(size_t) t.shard];
}

inline void ssd_pread(int fd, char* dst, size_t n, uint64_t off) {
    size_t got = 0;
    while (got < n) {
        const ssize_t k = ::pread(fd, dst + got, n - got, (off_t) (off + got));
        if (k < 0 && errno == EINTR) continue;
        if (k <= 0)
            throw std::runtime_error("FNX_SSD_EXPERTS: read at " + std::to_string(off + got) + ": " + (k < 0 ? std::strerror(errno) : "end of file"));
        got += (size_t) k;
    }
}
inline const SsdLayer& ssd_layer(int dev, int g) {
    const std::vector<SsdLayer>& v = g_ssd_layers[dev & (SSD_DEVS - 1)];
    if ((size_t) (g / NEXP) >= v.size() || v[(size_t) (g / NEXP)].fd_gate < 0 || v[(size_t) (g / NEXP)].fd_down < 0)
        throw std::runtime_error("FNX_SSD_EXPERTS: device " + std::to_string(dev) + " has no file places for expert " + std::to_string(g));
    return v[(size_t) (g / NEXP)];
}
inline void ssd_count(size_t bytes, std::chrono::steady_clock::time_point t0) {
    g_ssd_reads.fetch_add(1, std::memory_order_relaxed);
    g_ssd_bytes.fetch_add(bytes, std::memory_order_relaxed);
    g_ssd_ns.fetch_add((unsigned long long) std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count(),
                       std::memory_order_relaxed);
}
// expert g's gate|up part for the card on device dev into dst (2 gu_half bytes: the slot's layout)
inline void ssd_read_gu(int dev, int g, char* dst) {
    const auto t0 = std::chrono::steady_clock::now();
    const SsdLayer& s = ssd_layer(dev, g);
    const uint64_t e = (uint64_t) (g % NEXP);
    ssd_pread(s.fd_gate, dst, s.gu_half, s.gate0 + e * s.gu_mat + s.gu_skip);
    ssd_pread(s.fd_up, dst + s.gu_half, s.gu_half, s.up0 + e * s.gu_mat + s.gu_skip);
    ssd_count(2 * s.gu_half, t0);
}
// its down part (dn_rows dn_take bytes): the whole matrix read into tmp, the card's columns of each row taken from it
inline void ssd_read_dn(int dev, int g, char* dst, std::vector<char>& tmp) {
    const auto t0 = std::chrono::steady_clock::now();
    const SsdLayer& s = ssd_layer(dev, g);
    const uint64_t at = s.down0 + (uint64_t) (g % NEXP) * s.dn_mat;
    if (s.dn_skip == 0 && s.dn_take == s.dn_row) {
        ssd_pread(s.fd_down, dst, s.dn_mat, at);
    } else {
        tmp.resize(s.dn_mat);
        ssd_pread(s.fd_down, tmp.data(), s.dn_mat, at);
        for (int r = 0; r < s.dn_rows; ++r) std::memcpy(dst + (size_t) r * s.dn_take, tmp.data() + (size_t) r * s.dn_row + s.dn_skip, s.dn_take);
    }
    ssd_count(s.dn_mat, t0);
}

// fn(i, tmp) for i in [0, n) on up to `threads` threads (each its own tmp); the first error is thrown once all are done
inline void ssd_parallel(size_t n, int threads, const std::function<void(size_t, std::vector<char>&)>& fn) {
    if (n == 0) return;
    const int t = (int) std::min<size_t>(n, (size_t) std::max(1, threads));
    std::atomic<size_t> next{0};
    std::exception_ptr err;
    std::mutex mu;
    auto run = [&] {
        std::vector<char> tmp;
        try {
            for (size_t i; (i = next.fetch_add(1)) < n;) fn(i, tmp);
        } catch (...) {
            std::lock_guard<std::mutex> g(mu);
            if (!err) err = std::current_exception();
            next = n;
        }
    };
    std::vector<std::thread> th;
    for (int k = 1; k < t; ++k) th.emplace_back(run);
    run();
    for (auto& x : th) x.join();
    if (err) std::rethrow_exception(err);
}

// experts to their slots in VRAM: each one's gate|up and down destinations and bytes (the layer's strides)
struct SsdFill {
    int g;
    char* gu;
    char* dn;
    size_t gs, ds;
};
// read by ssd_threads() threads into page-locked buffers, 64 experts at a time in two halves, each half copied on st while the
// next is read; returns once every copy has landed. The card's device must be current
inline void ssd_fill(int dev, const std::vector<SsdFill>& jobs, cudaStream_t st) {
    if (jobs.empty()) return;
    int prev = 0;
    cudaGetDevice(&prev);
    if (cudaSetDevice(dev) != cudaSuccess) throw std::runtime_error("FNX_SSD_EXPERTS: fill: cudaSetDevice");
    struct Back {
        int d;
        ~Back() { cudaSetDevice(d); }
    } back{prev};
    constexpr size_t BATCH = 64;
    size_t most = 0;
    for (const SsdFill& j : jobs) most = std::max(most, j.gs + j.ds);
    most = (most + 255) & ~(size_t) 255;
    char* buf = nullptr;
    if (cudaHostAlloc((void**) &buf, 2 * BATCH * most, cudaHostAllocPortable) != cudaSuccess)
        throw std::runtime_error(std::string("FNX_SSD_EXPERTS: page-locked buffers for a fill: ") + cudaGetErrorString(cudaGetLastError()));
    cudaEvent_t ev[2] = {};
    bool used[2] = {false, false};
    try {
        for (cudaEvent_t& e : ev)
            if (cudaEventCreateWithFlags(&e, cudaEventDisableTiming) != cudaSuccess) throw std::runtime_error("FNX_SSD_EXPERTS: fill event");
        int half = 0;
        for (size_t b0 = 0; b0 < jobs.size(); b0 += BATCH, half ^= 1) {
            const size_t n = std::min(BATCH, jobs.size() - b0);
            char* base = buf + (size_t) half * BATCH * most;
            if (used[half] && cudaEventSynchronize(ev[half]) != cudaSuccess) throw std::runtime_error("FNX_SSD_EXPERTS: fill copy");
            ssd_parallel(n, ssd_threads(), [&](size_t k, std::vector<char>& tmp) {
                const SsdFill& j = jobs[b0 + k];
                ssd_read_gu(dev, j.g, base + k * most);
                ssd_read_dn(dev, j.g, base + k * most + j.gs, tmp);
            });
            for (size_t k = 0; k < n; ++k) {
                const SsdFill& j = jobs[b0 + k];
                if (cudaMemcpyAsync(j.gu, base + k * most, j.gs, cudaMemcpyHostToDevice, st) != cudaSuccess ||
                    cudaMemcpyAsync(j.dn, base + k * most + j.gs, j.ds, cudaMemcpyHostToDevice, st) != cudaSuccess)
                    throw std::runtime_error(std::string("FNX_SSD_EXPERTS: fill copy: ") + cudaGetErrorString(cudaGetLastError()));
            }
            if (cudaEventRecord(ev[half], st) != cudaSuccess) throw std::runtime_error("FNX_SSD_EXPERTS: fill event");
            used[half] = true;
        }
        if (cudaStreamSynchronize(st) != cudaSuccess)
            throw std::runtime_error(std::string("FNX_SSD_EXPERTS: fill: ") + cudaGetErrorString(cudaGetLastError()));
    } catch (...) {
        cudaStreamSynchronize(st);
        for (cudaEvent_t e : ev)
            if (e) cudaEventDestroy(e);
        cudaFreeHost(buf);
        throw;
    }
    for (cudaEvent_t e : ev) cudaEventDestroy(e);
    cudaFreeHost(buf);
}

// one line of what was read since the last line (the copier at idle, a prompt pass, a fill): parts, MiB, ms of reading
inline void ssd_report(const char* what) {
    static std::mutex mu;
    static unsigned long long r0 = 0, b0 = 0, n0 = 0;
    std::lock_guard<std::mutex> g(mu);
    const unsigned long long r = g_ssd_reads.load(), b = g_ssd_bytes.load(), n = g_ssd_ns.load();
    if (r == r0) return;
    std::fprintf(stderr, "fnx ssd: %s: %llu parts read, %.0f MiB, %.0f ms of reading (%.0f MiB/s a reader)\n", what, r - r0, (b - b0) / 1048576.0,
                 (n - n0) / 1e6, n > n0 ? (b - b0) / 1048576.0 / ((n - n0) / 1e9) : 0.0);
    r0 = r, b0 = b, n0 = n;
}

}  // namespace fnx
