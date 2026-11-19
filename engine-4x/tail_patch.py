#!/usr/bin/env python3
# UNVERIFIED PROOF OF CONCEPT: nobody has run this on four cards (engine-4x/README.md).
#
# tail_patch.py (2026-11-19, the 4-card proof of concept): the prompt path's down kernel for an FF slice that is not a whole
# number of 64 columns. Four cards cut the experts' 640 FF columns in slices of 160 (2.5 x 64); engine/'s p100_moe_mode refuses
# anything but multiples of 64. This is Kenzu's change for three P100 (224/224/192 columns), github.com/Kenzu/
# qwen3.8-flash-next-2x-p100, branch x99-p100, commit c43e4b8 ("Run GSQ-RCO IQ3_S with per-expert tensor slices on three Pascal
# GPUs"; MIT, as this repository): mode 5's tile (k_job_v64) gets a TAIL variant that loads zeros past the slice's last column, and
# a slice with a 32-column tail must run on mode 5 (down: FNX_PF_JOBV 5, 6 or 7, our templates' 7) with one of the four weight
# types the load converts down experts to. Kenzu's GPU test (engine/tools/p100_tail_test.cu on that branch) compares compact and
# zero-padded slices for those four types at widths 32 to 320, 160 among them, 1 to 96 tokens: 392 cases passed on their P100s.
# We have not run it. Run in the engine's source folder (the one holding src/).
import sys
from pathlib import Path

S = Path("src")
if not (S / "p100_moe.cu").exists():
    sys.exit("tail_patch.py: run it in the folder that holds src/")


def rep(name, old, new, count=1):
    p = S / name
    t = p.read_text(encoding="utf-8")
    n = t.count(old)
    if n != count:
        sys.exit(f"tail_patch.py: {name}: {n} matches, want {count}:\n{old[:300]}")
    p.write_text(t.replace(old, new), encoding="utf-8")


if "TAIL" in (S / "p100_moe.cu").read_text(encoding="utf-8"):
    sys.exit("tail_patch.py: applied already")

rep("p100_moe.cu",
    "template <int TY, int NT, int RL, int TL, bool PF, int NBX = k4NB>\n"
    "__device__ __forceinline__ void jobv_tile(",
    "template <int TY, int NT, int RL, int TL, bool PF, int NBX = k4NB, bool TAIL = false>   // TAIL: tail_patch.py (Kenzu, c43e4b8)\n"
    "__device__ __forceinline__ void jobv_tile(")
rep("p100_moe.cu",
    "                uint32_t rw[RunWords<TY>::n];\n"
    "                run_load_s<TY>(gp, run0 + dj, rw);\n"
    "                uint4 o[4];\n"
    "                jobv_finish<TY, NT>(tab, t, rw, o);\n"
    "#pragma unroll\n"
    "                for (int q = 0; q < 4; ++q) sw[dl * KU + 4 * dj + q] = o[q];\n",
    "                uint32_t rw[RunWords<TY>::n];\n"
    "                uint4 o[4];\n"
    "                if (!TAIL || (run0 + dj) * 4 < nu) {\n"
    "                    run_load_s<TY>(gp, run0 + dj, rw);\n"
    "                    jobv_finish<TY, NT>(tab, t, rw, o);\n"
    "                } else {   // past the slice's last column: zeros\n"
    "#pragma unroll\n"
    "                    for (int q = 0; q < 4; ++q) o[q] = make_uint4(0, 0, 0, 0);\n"
    "                }\n"
    "#pragma unroll\n"
    "                for (int q = 0; q < 4; ++q) sw[dl * KU + 4 * dj + q] = o[q];\n")
rep("p100_moe.cu",
    "                if (xi[q] >= 0) v[q] = __ldg(X + std::size_t(xi[q]) * ldx_u + u0 + (tid & (KU - 1)));\n",
    "                if (xi[q] >= 0) v[q] = (!TAIL || u0 + (tid & (KU - 1)) < nu)\n"
    "                    ? __ldg(X + std::size_t(xi[q]) * ldx_u + u0 + (tid & (KU - 1))) : make_uint4(0, 0, 0, 0);\n")
rep("p100_moe.cu",
    "template <int TY, int NT>\n"
    "__launch_bounds__(NT, 2)\n"
    "__global__ void k_job_v64(JobArgs a) {\n",
    "template <int TY, int NT, bool TAIL = false>\n"
    "__launch_bounds__(NT, 2)\n"
    "__global__ void k_job_v64(JobArgs a) {\n")
for rl, tl, cond in [(4, 6, "if (nt > 72)"), (2, 9, "else if (nt > 48)"), (2, 6, "else if (nt > 36)"), (1, 9, "else if (nt > 24)"),
                     (1, 6, "else if (nt > 12)"), (1, 3, "else")]:
    rep("p100_moe.cu",
        f"    {cond} jobv_tile<TY, NT, {rl}, {tl}, false, 64>(a, sm, jg, 0, tend);\n",
        f"    {cond} jobv_tile<TY, NT, {rl}, {tl}, false, 64, TAIL>(a, sm, jg, 0, tend);\n")
rep("p100_moe.cu",
    "            k_job_v64<TY, 256><<<dim3(unsigned((a.N + 63) / 64), njobs_cap), 256, kJobV64Smem<TY, 256>, s>>>(a);\n",
    "            if (a.K % 64) k_job_v64<TY, 256, true><<<dim3(unsigned((a.N + 63) / 64), njobs_cap), 256, kJobV64Smem<TY, 256>, s>>>(a);\n"
    "            else k_job_v64<TY, 256, false><<<dim3(unsigned((a.N + 63) / 64), njobs_cap), 256, kJobV64Smem<TY, 256>, s>>>(a);\n")
rep("p100_moe.cu",
    "    if (A.n_in % 64) throw std::runtime_error(\"p100 moe: columns not in units of 64\");\n",
    "    if (A.n_in % 32) throw std::runtime_error(\"p100 moe: columns not in units of 32\");   // tail_patch.py: 32-column tails on mode 5\n")
rep("p100_moe.cu",
    "    if (mode == 7) mode = A.n_in <= 640 ? 5 : 7;   // jobv7: down as 6, gate|up on k_job_v7 (IQ2_S on mode 4)\n",
    "    if (mode == 7) mode = A.n_in <= 640 ? 5 : 7;   // jobv7: down as 6, gate|up on k_job_v7 (IQ2_S on mode 4)\n"
    "    if (A.n_in % 64) {   // tail_patch.py (Kenzu, c43e4b8): a slice of 64s and 32 columns, mode 5's TAIL tile only\n"
    "        if (mode != 5) throw std::runtime_error(\"p100 moe: a 32-column tail needs the down kernel of FNX_PF_JOBV 5, 6 or 7\");\n"
    "        if (A.type != T_Q4_0 && A.type != T_IQ4_NL && A.type != T_Q5_1 && A.type != T_Q8_0)\n"
    "            throw std::runtime_error(\"p100 moe: a 32-column tail on an expert type the TAIL tile does not take\");\n"
    "    }\n")
print("tail_patch.py: 32-column tails on the prompt path's down kernel (Kenzu, c43e4b8)")
