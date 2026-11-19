# The 32 GB mode's engine

The fast UD-Q4_K_XL on a machine with 32 GB of RAM ([docs/GUIDE.md, "32 GB of RAM"](../docs/GUIDE.md#32-gb-of-ram)) runs an
engine of its own, so the engine every other profile runs (`engine/`, `build/`) is not touched by it:

```bash
bash build.sh --32g      # build-32g/split/strata-split-2
bash setup.sh /data/flash-next-q4-32 --models /data/flash-next-q4/models --model q4_k_xl --fast --ram 32
```

`build.sh --32g` copies `engine/` to `build-32g/engine-32g`, applies the two patches below to the copy, and builds it into
`build-32g`. The profile `setup.sh --ram 32` (or `profile/fill_profile.py --ram 32`) writes names that engine.

| File | What it is |
|---|---|
| `ssd_patch.py` | `FNX_SSD_EXPERTS`: no routed expert gets a page-locked RAM copy; its host copy is its bytes in the model's files, read when a card needs it (by each card's copier while writing, `FNX_SSD_AHEAD` layers ahead while reading a prompt). Copies `ssd.hpp` from beside itself into `src/` |
| `ssd3_patch.py` | `FNX_SSD_RAM_GIB`: the RAM tier, page-locked, exclusive with the cards' expert slots (an evicted expert is copied down into it before its slot is written), warmed at the load, read by the prompt path. Runs after `ssd_patch.py` |
| `ssd.hpp` | the reads from the files, the slot fills, the switches' defaults |

Both patches check every edit they make (the build stops if one does not match), and with `FNX_SSD_EXPERTS` unset the
patched engine takes none of the new paths. They apply to `engine/`'s sources of 19 Nov 2026 (SHA-256 in
[docs/PROVENANCE.md](../docs/PROVENANCE.md)); our build of the result, `4b5107bf`, is what the 32 GB mode was measured on.
The switches are in [docs/SWITCHES.md, "The 32 GB mode"](../docs/SWITCHES.md#the-32-gb-mode).
