# Four Tesla P100 behind one PCIe switch: an unverified proof of concept

> **UNVERIFIED PROOF OF CONCEPT. Nobody has run this on four cards.** We have one machine with two P100, so this build has
> never loaded a model, never produced a token and never been timed. What we checked: `bash build.sh --4x` compiles
> it for sm_60 (our server, CUDA 12.4, g++-13, 19 Nov 2026; `strata-split-4`, SHA-256 `2d9cef00dd7bae49`), and the new
> gather's flags were simulated on the CPU for 2 to 8 cards (`engine-4x/sim_gather.py`). The two-card build is unchanged:
> `engine/` is not touched and `bash build.sh` passes the same values as before. Everything below
> about speed is an estimate from arithmetic. If you run it, the section "Turning it into a verified build" says what to
> send back.

This is a build of our two-card engine (a Tesla P100 port of Strata's split engine, which is built from
[sFNX](https://github.com/johnfink8/sfnx)) for one particular machine:

- four Tesla P100 16 GB (sm_60), each on PCIe 3.0 x8,
- all four behind one PCIe switch,
- the switch on the chipset by PCIe 3.0 x4, and the chipset on the processor by its own link (DMI or similar), which the
  disks and the network share,
- ISTA-DASLab's GSQ-RCO IQ3_S (the model this repository runs fastest).

It lives in [`engine-4x/`](../engine-4x/README.md) and `build.sh --4x`. `engine/` and the two-card build are not changed.

## What limits this machine

Everything that goes between the cards and RAM crosses one PCIe 3.0 x4 link: about 3.5 GB/s in each direction, **for all
four cards together**. Our two-card machine has 12 GB/s for each card (24 together), so this one has about 7 times less.
The cards themselves are close to each other (x8 to the switch, about 6.5 GB/s each), and the switch can pass traffic
between them without the x4 link, but only with peer access, which many machines do not allow (see below).

What crosses that link in our engine:

| Traffic | Two cards (ours) | Four cards here | What this build does about it |
|---|---|---|---|
| Experts that are not in VRAM, copied in when a token needs them | about 60 % of IQ3_S's 48 GiB of experts live only in RAM | about 10 % or less (estimate: about 11 GiB of experts a card) | the profile keeps as many experts on the cards as fit (`FNX_EXPERT_RAM_GIB=3`): four cards hold twice the VRAM |
| A decode window's sums (after every layer's attention and experts, 96 a token) | host memory, each card's own x16 | host memory over the x4 link: about 11 MiB a window row, about 3 ms (estimate) | peer access when the machine allows it: card to card through the switch, nothing on the x4 link |
| A prompt chunk's sums and attention exchange | host memory | host memory over the x4 link: about 3.7 MiB a prompt token, about 1.1 ms (estimate) | the direct gather (`FNX_PF_GATHER=direct`): 25 % fewer bytes into the cards than the default ring |
| The lookup table's rows (26.8 GiB, `per_layer_token_embd`) | RAM | RAM | kept in RAM (`FNX_PLE_RAM=1`): from the disk it would cross the chipset's link too |

A miss costs all four cards their slice of the expert at once: about 0.5 MiB each (IQ3_S; 1 MiB on two cards), 2 MiB over
the shared link. At 1 % of a token's 480 expert uses missed, that is about 10 MiB, about 3 ms a token. So on this machine
the experts must stay on the cards, and that decides the profile.

## What is changed, and why

### The engine (`engine-4x/`, `bash build.sh --4x`)

`build.sh --4x` copies `engine/` to `build-4x/engine-4x`, applies three patches and builds `strata-split-4`
(`STRATA_SPLIT_CARDS=4`). The engine was written for 2, 3, 4, 6 or 8 cards (sFNX, then Strata, ran it on three and four
cards); our P100 additions assumed two in three places, and four cards need one kernel change:

1. **`pf_gather` (`FNX_PF_SP`), `ncard4_patch.py`.** The exchange that trades the rows each card reads `hc` for was
   written for two cards (a `static_assert`). Now each card's finished part goes once to its own host rows and every other
   card copies it from there, with a flag per sender and a "read" flag back. All host rows are in the same RAM, so a part
   crosses the x4 link upwards once, not once a receiver. On two cards it moves the same bytes in the same order as
   before. `FNX_PF_SP` stays **off** in the 4-card profiles: it needs the ring gather (`FNX_PF_GATHER=ring`) and moves
   more bytes over the x4 link (the mixed rows in fp32) than its saved compute is worth here. Its parts are quarters on four
   cards, and `sp_algo`'s cuBLAS choices were checked for halves only, so with it on the bits may differ from it off (not
   checked).
2. **Edge0-145B, `ncard4_patch.py`.** Its latent experts are cut in NVFP4's 64-value blocks and `FNX_LAT_SPLIT` in two
   halves; four cards' FF slices are 160 wide. The engine refuses `FNX_LATENT` on more than two cards with a plain message
   at the load, before anything is placed. Run Edge0 with the two-card build.
3. **The down kernel's 32-column tail, `tail_patch.py`.** Four cards cut each expert's 640 FF columns in slices of 160
   (2.5 x 64); the prompt path's down kernel took multiples of 64 only. This is Kenzu's change for three P100
   ([Kenzu/qwen3.8-flash-next-2x-p100](https://github.com/Kenzu/qwen3.8-flash-next-2x-p100), branch `x99-p100`, commit
   `c43e4b8`), applied unchanged: mode 5's tile loads zeros past the slice's last column. Kenzu's GPU test compares compact
   and zero-padded slices for the four weight types the load converts down experts to, widths 32 to 320 (160 among them),
   1 to 96 tokens: 392 cases passed on their P100s. We have not run it again.
4. **Peer access on NVIDIA (`FNX_PEER`), `peer_patch.py` and `cuda_peer.hpp`.** On AMD the engine already pushes the
   decode windows' sums (`k_ring_a2a_push`) and attention merge (`k_att_merge_a2a`) into the other cards' memory when every
   pair passes a start probe, else it uses the host exchange. These kernels are plain CUDA; only the decision and the
   mailboxes were AMD-only. The patch adds them on NVIDIA:
   - every card's IOMMU group is read first; a translated domain (`DMA`, `DMA-FQ`) faults every peer write, so then there is
     no probe, only the host exchange;
   - `cudaDeviceCanAccessPeer` for every pair, then `cudaDeviceEnablePeerAccess`;
   - every directed pair: 4 MiB stored by the sender into the receiver's memory, compared in full by the receiver, the
     stores timed, and 200 flag round trips. Nothing reads a peer's memory, the probe included: the engine only pushes, and a
     P2P read the platform does not route can end the CUDA context;
   - any failure: peer access off again on every pair, the host exchange for the whole run, one line in the log saying why.

   The prompt path still exchanges through host memory, with or without peer access (see "Not done").

`build.sh` refuses a two-card binary that can call a peer-access function, as before; for `--4x` it checks that the
binary can (the patch took) and lists the functions.

### The profiles (`engine-4x/*.template.json`, `engine-4x/fill_profile_4x.py`)

Our IQ3_S 512k profile with these changes:

| Switch | Two cards | Four cards | Why |
|---|---|---|---|
| `gpu` | `0,1` | `0,1,2,3` | |
| `FNX_EXPERT_RAM_GIB` | 29.0 | 3.0 | only the coldest 3 GiB of experts stay in RAM only; the rest size the cards' slots. If they do not fit, the engine moves more of the coldest to RAM by itself at the load and says so (`VRAM short by ... more of the coldest experts to RAM`) |
| `FNX_PF_GATHER` | unset (ring) | `direct` | the prompt path's all-gather: each finished part stored straight into every card's host rows. Per chunk sum about 6 times the chunk's f16 rows go into the cards over the x4 link instead of 8 with the ring's forwarding (its host-to-host copies cross the link twice). Strata's AMD default for four cards; on NVIDIA we have not run it (Kenzu's three cards ran the ring): if reading misbehaves, `ring` is the fallback |
| `FNX_PF_SP` | 1 | 0 | see above |
| `FNX_PLE_RAM` | 1 (default) | 1, written out | the lookup table in RAM, not read from a disk behind the same chipset |
| `FNX_PEER` | (none) | 1 | try peer access, fall back by itself; `0` forces the host exchange |
| `--max-context` | 524288 (YaRN x2) | 262144 by default, no YaRN | the K/V pool grows into the experts' slots as a conversation gets longer (`FNX_ARENA`): at 512k up to about 3 GiB a card (a quarter of what experts get), at 256k half of that. Short conversations keep the same experts either way. `--context 512k` gives our production's 524,288 with YaRN, `--context 128k` 131,072 |

Everything else is our production's (`FNX_PREFETCH_WANT=6`, `FNX_PF_JOBV=7`, `FNX_PF_TAIL_MIB=3000` ...). Not tuned for
this machine, because we cannot measure on it; "Expectations" names the switches to try first.

### Not done

- **The prompt path over peer access.** Its sums and attention exchange go through the copy engines into host memory and
  out again, with flags in host memory, even when peer access works. Pointing them at device memory on the receiving card
  would keep them off the x4 link, but then each flag must reach the receiver after the data on another path, which needs
  care and a machine to test it on. If peer access works on your machine and reading is the bottleneck, this is the next step.
- **UD-Q4_K_XL and Edge0-145B.** UD-Q4_K_XL's experts are 72 GiB, so about 40 % would stay in RAM behind the x4 link;
  Edge0 is refused (above).
- **`setup.sh`.** It still writes the two-card profile and wants the two-card engine; the 4-card profile comes from
  `engine-4x/fill_profile_4x.py`.

## Build and run

```bash
bash build.sh --4x                       # build-4x/split/strata-split-4 (about 20 minutes with 8 jobs)
bash build.sh                            # the two-card engine too: setup.sh wants it, and it is the reference below
bash setup.sh /data/flash-next           # models, Strata's server, the tokenizer, the MTP layer, a two-card profile
python3 engine-4x/fill_profile_4x.py --models /data/flash-next/models --mtp /data/flash-next/prep/mtp-rt \
    --tokenizer /data/flash-next/prep/tokenizer --logs /data/flash-next/logs \
    --vision-exe /data/flash-next/strata-v0.2.9/build-vision/bin/strata-vision --out /data/flash-next/profile-4x.json
bash run.sh /data/flash-next --config /data/flash-next/profile-4x.json
```

Without `--vision-exe` the profile runs without pictures. RAM: our two-card IQ3_S profile holds about 85 GiB once loaded;
four cards add each card's page-locked exchange buffers, so expect about 90 GiB (estimate). 96 GB is tight, 128 GB is
comfortable; below that see "Less RAM".

## Check the machine first

1. **The topology:** `nvidia-smi topo -m`. Between every two cards it should say `PIX` (one switch between them) or `PXB`
   (several switches). `PHB`, `NODE` or `SYS` means their traffic goes up through a host bridge or a processor, and peer
   access through the switch is not what will happen.
2. **What the driver allows:** `nvidia-smi topo -p2p w` (writes; the engine only writes) and `nvidia-smi topo -p2p r`.
   `OK` for every pair is needed for peer access. `CNS` ("chipset not supported") is common behind a chipset; then the
   engine uses the host exchange by itself.
3. **The links between the cards, measured:** CUDA's sample `p2pBandwidthLatencyTest`
   ([cuda-samples](https://github.com/NVIDIA/cuda-samples), `Samples/5_Domain_Specific/p2pBandwidthLatencyTest`). With peer
   access on, unidirectional bandwidth between two cards behind one switch on Gen3 x8 should be near 6 GB/s, and the
   latency a few microseconds. If the "P2P enabled" rows are no faster than the "disabled" ones, the traffic is going up
   to the root.
4. **ACS on the switch:** with Access Control Services on, the switch's downstream ports send peer traffic up to the root
   (through the x4 link), and the chipset may drop it. As root:
   `lspci -vvv | grep -E '^[0-9a-f]|ACSCtl'`. On the switch's downstream ports `ACSCtl` should show `ReqRedir-` and
   `CmpltRedir-`. Some BIOSes have a switch for it; otherwise `setpci -s <port> ECAP_ACS+6.w=0000` turns it off on one port
   until the next boot. ACS is there to isolate devices from each other under an IOMMU, so turn it off only with the IOMMU
   off or in passthrough (next point).
5. **The IOMMU off or in passthrough:** `cat /proc/cmdline` should hold `iommu=pt` (or `intel_iommu=off` /
   `amd_iommu=off`), and `cat /sys/kernel/iommu_groups/*/type` should print `identity` (or there are no groups). A
   translating IOMMU faults every peer write (`dmesg`: `DMAR: ... PTE Write access is not set`); the engine reads this
   itself and then does not try.
6. **The whole machine:** `python3 tune/check.py --cards 4 --models /data/flash-next/models` (about a minute, with the
   cards free). It reads the cards' PCIe paths, RAM, the IOMMU, `nvidia-smi topo`, and copies to each card alone and to
   all four at once. All four at once shows the shared link: about 3.2-3.5 GB/s together on PCIe 3.0 x4.
7. **At the engine's start**, its log (`logs/strata-split-4.log`) says which exchange it took:
   - `fnx: Peer push between the cards is on (all 12 directed pairs: 4 MiB stored and compared in full on the receiving card, A to B GB/s, flag round trip at most C us).`
     followed by `fnx: engine-4x: the decode windows' sums and attention merge go card to card; ...`
   - or `fnx: The cards cannot push into each other's memory here (<why>): using the host exchange for this run.` followed
     by `fnx: engine-4x: the cards exchange through page-locked host memory (no peer access)`.
   - and how many experts went where: `fnx: N of 24576 routed experts (X GiB) in RAM, ranked by ...`.

## Expectations (estimates, not measurements)

Nobody has run this. These numbers come from the byte counts above and from our two-card machine's measurements
(IQ3_S 512k: writing 66.5 tok/s, edits 110.7 / 115.8, reading a 38k prompt 1,012 tok/s); they may be wrong by a lot.

| | Two cards, x16 each (ours, measured) | Four cards behind x4, host exchange (estimate) | Same with peer access (estimate) |
|---|---|---|---|
| Experts in VRAM | about 40 % of 48 GiB | about 90 % or more | the same |
| Fresh writing | 66.5 tok/s | 50-80 tok/s | 60-95 tok/s |
| Reading a long prompt | 1,012 tok/s | 450-850 tok/s (the x4 link caps it near 900) | the same: the prompt path does not use peer access |

Why: each card does half the expert work of a two-card setup, and far fewer experts are copied. Against that, without peer
access every decode window's 96 sums cross the shared link (about 3 ms a window row, plus their latency through switch and
chipset), and four cards wait for each other at every one of them. Reading a prompt is bound by the x4 link: about 3.7 MiB
of exchange a prompt token into the cards.

Switches to try first, one at a time, with `bench/bench.py`:
- `FNX_PREFETCH_WANT` 2 or 3 (guessed next-layer copies share the x4 link with the real misses; Kenzu used 2 on three cards);
- `FNX_ONEHOP=0` (the ring for the decode sums: about 25 % fewer bytes into the cards than the one-hop sum, but six hops in a
  row instead of one; may win on long windows without peer access);
- `FNX_PF_GATHER=ring` (if reading is wrong or stalls);
- `FNX_PEER=0` (to compare with the host exchange when peer access is on);
- `--context 512k` against the default 256k, deep into a long conversation.

## Less RAM

Untested starting point for 64 GB: with four cards most experts can live on the cards without a RAM copy.
`FNX_EXPERT_DYN_GIB=18` keeps only the coldest 18 GiB of the experts in VRAM dynamic (with a RAM copy) and the rest
permanent without one, so RAM holds `FNX_EXPERT_RAM_GIB` + 18 GiB of experts instead of all 48 ([GUIDE.md, "64 GB of RAM"](GUIDE.md#64-gb-of-ram),
explains the switch on two cards). The K/V pool and the prompt path borrow only dynamic slots, so 18 is about the least for
256k (estimate: about 1.5 GiB of K/V and 3 GiB of prompt buffers a card). If the load stops with `FNX_EXPERT_DYN_GIB: VRAM short by ...`, raise `FNX_EXPERT_RAM_GIB` by what it says. Keep
`FNX_PLE_RAM=1` if it fits; the lookup table from the disk crosses the chipset too.

## Turning it into a verified build

On the four-card machine, after the checks above:

1. Run the two-card engine on two of the four cards (`bash build.sh`, `bash setup.sh DATA`, `"gpu": "0,1"`) with
   `"FNX_DUMP_FIRST_LOGITS": "/tmp/logits-2"` added to its profile's `env`, send it a few requests at temperature 0, stop it.
2. Run the 4-card engine with `"FNX_DUMP_FIRST_LOGITS": "/tmp/logits-4"`, send the same requests in the same order.
3. `python3 engine-4x/compare_logits.py /tmp/logits-2 /tmp/logits-4`: the top token should be the same and the KL small
   (the cards add their parts in another order, so the logits are not bit-equal). Do it once with `FNX_PEER=0` and once
   with peer access, if the machine allows it.
4. `python3 bench/bench.py ...` ([GUIDE.md, "Benchmark and compare"](GUIDE.md#benchmark-and-compare)) for the speeds.

Send back the start of `logs/strata-split-4.log` (the `fnx:` lines), `tune/check.py --cards 4`'s report, the comparison and
the speeds. Until then this is a proof of concept.

## Credits

- [sFNX](https://github.com/johnfink8/sfnx) by John Fink (MIT): the split engine, written for any count of cards, with
  peer access on V100s.
- [Strata](https://github.com/Niko1221/Strata) (MIT): the host exchange, the one-hop sums, the direct gather and the AMD
  peer push this build turns on for NVIDIA.
- Kenzu ([Kenzu/qwen3.8-flash-next-2x-p100](https://github.com/Kenzu/qwen3.8-flash-next-2x-p100), `x99-p100`): the down
  kernel's 32-column tail and its GPU test, from their three-P100 port.
