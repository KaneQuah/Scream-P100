#!/usr/bin/env python3
# UNVERIFIED PROOF OF CONCEPT: nobody has run the 4-card engine on four cards (engine-4x/README.md).
#
# sim_gather.py: a CPU check of ncard4_patch.py's pf_gather flags, run on 19 Nov 2026 for 2, 3, 4, 6 and 8 cards, 400 random
# interleavings each: "all interleavings fine". With the slots swapped it fails from three cards on, so it does catch a wrong flag.
#
# Simulates ncard4_patch.py's pf_gather protocol: one in-order stream per card (waits block, copies/flag writes run when reached),
# a random interleaving of the streams, several gathers in a row (both sets reused). Checks: no deadlock, every card ends each
# gather with every part's value of that gather, and no sender overwrites its host rows before every receiver copied them.
import random
def slot(n, g, d): return ((g - d - 1) % n + n) % n
def run(n, gathers, seed):
    rnd = random.Random(seed)
    SL, SET = n - 1, 2 * (n - 1)
    flags = [[0] * (4 * (n - 1)) for _ in range(n)]
    gown = [[None, None] for _ in range(n)]
    base = [[None] * n for _ in range(n)]          # base[card][part]
    reads = {}                                     # (owner, set) -> readers still to copy the current content
    ops = [[] for _ in range(n)]
    for seq in range(1, gathers + 1):
        st = seq & 1
        for g in range(n):
            own = (g + 1) % n
            o = ops[g]
            o.append(("produce", seq))               # this card's finished part of gather seq appears in base (stream order)
            if seq > 2:
                for j in range(SL): o.append(("wait", g, st * SET + SL + j, seq - 2))
            o.append(("send", st, seq))
            for d in range(n):
                if d != g: o.append(("write", d, st * SET + slot(n, g, d), seq))
            for k in range(n):
                if k == g: continue
                o.append(("wait", g, st * SET + slot(n, k, g), seq))
                o.append(("recv", k, st, seq))
            for k in range(n):
                if k != g: o.append(("write", k, st * SET + SL + slot(n, g, k), seq))
            o.append(("check", seq))
    pc = [0] * n
    while True:
        ready = [g for g in range(n) if pc[g] < len(ops[g]) and not (ops[g][pc[g]][0] == "wait" and flags[ops[g][pc[g]][1]][ops[g][pc[g]][2]] < ops[g][pc[g]][3])]
        if not ready:
            if all(pc[g] == len(ops[g]) for g in range(n)): return True
            return f"deadlock at {pc}"
        g = rnd.choice(ready)
        op = ops[g][pc[g]]; pc[g] += 1
        if op[0] == "produce": base[g][(g + 1) % n] = (op[1], (g + 1) % n)
        elif op[0] == "send":
            st, seq = op[1], op[2]
            left = reads.get((g, st))
            if left: return f"card {g} overwrote set {st} (gather {seq}) before cards {left} read it"
            gown[g][st] = base[g][(g + 1) % n]; reads[(g, st)] = set(range(n)) - {g}
        elif op[0] == "write": flags[op[1]][op[2]] = max(flags[op[1]][op[2]], op[3])
        elif op[0] == "recv":
            k, st, seq = op[1], op[2], op[3]
            v = gown[k][st]
            if v != (seq, (k + 1) % n): return f"card {g} read {v} from card {k} set {st}, want {(seq, (k + 1) % n)}"
            base[g][(k + 1) % n] = v; reads[(k, st)].discard(g)
        elif op[0] == "check":
            if base[g] != [(op[1], p) for p in range(n)]: return f"card {g} after gather {op[1]}: {base[g]}"
bad = 0
for n in (2, 3, 4, 6, 8):
    for seed in range(400):
        r = run(n, 7, seed)
        if r is not True: print(n, seed, r); bad += 1; break
print("all interleavings fine" if not bad else f"{bad} failures")
