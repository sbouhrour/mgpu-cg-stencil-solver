#!/usr/bin/env bash
# Writes an Open MPI rankfile that binds rank i to CPU cores local to GPU i, read from sysfs
# (local_cpulist of the GPU's PCI device), one logical CPU per physical core. GPUs that share a NUMA
# node split its cores evenly, so no two ranks share a core: Open MPI 5 rejects overlapping bindings,
# Open MPI 4 accepts both forms.
#
# Why pin: a rank's host buffers are placed on the NUMA node of the core that first touches them.
# On an 8x A100 node with 8 NUMA nodes, 8 GPUs copying 128 KiB to the host at once reached
# 0.65 to 1.55 GB/s per GPU unpinned and 7.25 GB/s pinned, and the 27-point 256^3 CG on 8 GPUs ran
# 526 ms unpinned against 327 ms pinned.
#
# Usage:  ./scripts/benchmarking/make_rankfile.sh > rankfile
#         RANKFILE=$PWD/rankfile ./scripts/benchmarking/comm_matrix.sh
# Exits 1 without output when a GPU reports no local CPU list (a VM hiding its NUMA layout).
set -uo pipefail
python3 - "$(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader)" <<'EOF'
import re
import sys

def expand(cpulist):
    cpus = []
    for part in cpulist.split(","):
        lo, _, hi = part.partition("-")
        cpus += range(int(lo), int(hi or lo) + 1)
    return cpus

def physical(cpus):
    # Rankfile slots name cores; a hyperthread sibling is not a core of its own
    keep = []
    for c in cpus:
        try:
            sib = open(f"/sys/devices/system/cpu/cpu{c}/topology/thread_siblings_list").read()
        except OSError:
            keep.append(c)
            continue
        if int(re.split(r"[,-]", sib.strip())[0]) == c:
            keep.append(c)
    return keep

def slot(cores):
    if cores == list(range(cores[0], cores[-1] + 1)):
        return f"{cores[0]}-{cores[-1]}"
    return ",".join(map(str, cores))

buses = [b.strip() for b in sys.argv[1].split("\n") if b.strip()]
lists = []
for b in buses:
    dev = b[-12:].lower()
    try:
        lists.append(open(f"/sys/bus/pci/devices/{dev}/local_cpulist").read().strip())
    except OSError:
        lists.append("")
if not all(lists):
    sys.exit(1)
groups = {}
for gpu, cl in enumerate(lists):
    groups.setdefault(cl, []).append(gpu)
slots = {}
for cl, gpus in groups.items():
    cores = physical(expand(cl))
    share = max(1, len(cores) // len(gpus))
    for k, gpu in enumerate(gpus):
        mine = cores[k * share:(k + 1) * share] or cores
        slots[gpu] = slot(mine)
for gpu in sorted(slots):
    print(f"rank {gpu}=localhost slot={slots[gpu]}")
EOF
