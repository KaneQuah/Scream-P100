# The 4-card engine (UNVERIFIED PROOF OF CONCEPT)

> **UNVERIFIED PROOF OF CONCEPT. Nobody has run this on four cards.** It is built for a machine we do not have: four Tesla
> P100 16 GB, each on PCIe 3.0 x8 behind one PCIe switch, the switch on the chipset by PCIe 3.0 x4. What it is, why, how to
> check that machine and what to expect: [docs/POC-4X.md](../docs/POC-4X.md).

```bash
bash build.sh --4x       # build-4x/split/strata-split-4; engine/ and build/ are not touched
python3 engine-4x/fill_profile_4x.py --models DATA/models --mtp DATA/prep/mtp-rt --tokenizer DATA/prep/tokenizer \
    --logs DATA/logs --out DATA/profile-4x.json
bash run.sh DATA --config DATA/profile-4x.json
```

`build.sh --4x` copies `engine/` to `build-4x/engine-4x`, applies the three patches below to the copy in this order, and
builds it for four cards (`STRATA_SPLIT_CARDS=4`). Every patch checks every edit it makes (the build stops if one does not
match).

| File | What it is |
|---|---|
| `ncard4_patch.py` | `pf_gather` (`FNX_PF_SP`) for any count of cards: each card's finished part goes once to its own host rows and the others copy it from there (two cards: the same bytes in the same order as before). Edge0-145B (`FNX_LATENT`, `FNX_LAT_SPLIT`) refused on more than two cards with a plain message at the load |
| `ff64_patch.py` | four cards cut the experts' FF width in 64-column units (192, 192, 128, 128), not 160 each: the first-token logits of the 160-column build were wrong in a hardware run (issue #1); applied first, so `tail_patch.py`'s tail path is not used |
| `tail_patch.py` | the prompt path's down kernel for FF slices of 160 columns (2.5 x 64): Kenzu's change for three P100 (github.com/Kenzu/qwen3.8-flash-next-2x-p100, commit `c43e4b8`), unchanged |
| `peer_patch.py` | `FNX_PEER` on NVIDIA: the decode windows' sums and attention merge pushed into the other cards' memory when every pair passes a start probe, else the host exchange; copies `cuda_peer.hpp` into `src/` |
| `prompt_peer_patch.py` | `FNX_PF_PEER=1` (off by default): the prompt path's chunk sums and attention partials go card to card into device buffers instead of page-locked host memory, flags stay in host memory; needs the peer probe to pass |
| `cuda_peer.hpp` | the IOMMU check, peer access and the start probe (from `engine/src/hip/peer.h`, the AMD version) |
| `flash-next-iq3s-4x-256k.template.json`, `flash-next-iq3s-4x-512k.template.json` | the IQ3_S profiles for four cards |
| `fill_profile_4x.py` | writes a run config from them through `profile/fill_profile.py` (`--context 256k`, the default, `512k` or `128k`) |
| `sim_gather.py` | a CPU simulation of the new gather's flags for 2 to 8 cards (no deadlock, every part right, none overwritten before it is read) |
| `compare_logits.py` | compares two runs' first-token logits (`FNX_DUMP_FIRST_LOGITS`), e.g. the 4-card engine against the 2-card one |

`tune/check.py --cards 4` looks at such a machine (all four cards, copies to all four at once, `nvidia-smi topo`).
