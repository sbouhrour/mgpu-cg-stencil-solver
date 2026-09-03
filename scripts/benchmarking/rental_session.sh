#!/usr/bin/env bash
# Runs a benchmark session on a rented instance, unattended, and collects everything.
#
# Three questions, in decreasing order of how much they need this machine:
#
#   B7  Does narrowing the stencil coefficients pay on datacenter hardware?
#       On an RTX 4060 the gain stops well short of the traffic ratio, because that part runs FP64
#       at a sixty-fourth of FP32, so the double-precision accumulation emerges as soon as the
#       coefficients stop dominating the traffic. A datacenter part runs FP64 at a half. Single GPU,
#       wall-clock only -- no hardware counters, so it is unaffected by a container marketplace
#       denying ncu.
#
#   B6  Is the coefficient-major (SoA) layout still ahead at solver level, multi-GPU?
#       Needs several GPUs.
#
#   B1  How much do the published numbers lose to being built at -O2 while AmgX is built at -O3?
#       The published figures are A100. A rebuild on any other part answers a different question.
#
# Usage:
#   ./scripts/benchmarking/rental_session.sh --only=b7          # B7 alone (the single-GPU session)
#   ./scripts/benchmarking/rental_session.sh --only=b7 --dry-run # local rehearsal, small and quick
#   ./scripts/benchmarking/rental_session.sh                     # everything, as before
#
#   --sizes="192 256"   grid sizes to sweep      --procs=N   separate processes per size
#   --reps=N            timed repetitions each   --gpus=N    cap on ranks for B6
set -uo pipefail
cd "$(dirname "$0")/../.."
OUT="${OUT_DIR:-out}"
mkdir -p "$OUT" matrix

ONLY=all
DRYRUN=0
SIZES_SET=""
PROCS="${PROCS:-5}"
REPS="${REPS:-11}"
MAXG_SET=""
for a in "$@"; do
    case "$a" in
        --only=*)  ONLY="${a#*=}" ;;
        --sizes=*) SIZES_SET="${a#*=}" ;;
        --procs=*) PROCS="${a#*=}" ;;
        --reps=*)  REPS="${a#*=}" ;;
        --gpus=*)  MAXG_SET="${a#*=}" ;;
        --dry-run) DRYRUN=1 ;;
        [0-9]*)    MAXG_SET="$a" ;;                 # positional max_gpus, kept for compatibility
        *) echo "unknown argument: $a"; exit 2 ;;
    esac
done

log() { printf '\n[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }

# --------------------------------------------------------------------------------------------------
# Machine
# --------------------------------------------------------------------------------------------------
command -v nvidia-smi >/dev/null || { echo "nvidia-smi absent"; exit 1; }
NGPU_AVAIL=$(nvidia-smi --list-gpus | wc -l)
MAXG="${MAXG_SET:-$NGPU_AVAIL}"
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' .')
VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
RAM_MB=$(awk '/MemAvailable/{printf "%d", $2/1024}' /proc/meminfo)

if [ "$DRYRUN" = 1 ]; then
    SIZES="${SIZES_SET:-128 192}"
    PROCS="${PROCS}"
else
    # 192 up: at 128 cubed the ratio is bimodal across processes on the reference consumer part,
    # spanning 0.78 to 1.88 -- the working set is too small to hold the kernel in one regime. The
    # sweep exists to show the ratio does not depend on size; it does not need to fill the card.
    # Adding 448 saturates an 80 GB device, but needs 64 GB of host RAM and doubles the session.
    SIZES="${SIZES_SET:-192 256 320 384}"
fi

# --------------------------------------------------------------------------------------------------
# Clocks
#
# The 4060 measurements carry a caveat the published page states plainly: the precision ratio moves
# between runs by more than a tenth, because the part's clock is managed against a shared power
# budget. A datacenter card can be pinned. Locking is attempted, never required: if it fails the
# session still runs and the dispersion across processes is reported instead of being assumed away.
# --------------------------------------------------------------------------------------------------
lock_clocks() {
    local maxsm maxmem
    maxsm=$(nvidia-smi --query-gpu=clocks.max.sm --format=csv,noheader,nounits | head -1)
    maxmem=$(nvidia-smi --query-gpu=clocks.max.mem --format=csv,noheader,nounits | head -1)
    if nvidia-smi -pm 1 >/dev/null 2>&1 && nvidia-smi -lgc "$maxsm" >/dev/null 2>&1; then
        nvidia-smi -lmc "$maxmem" >/dev/null 2>&1
        echo "  clocks LOCKED at sm=${maxsm} MHz mem=${maxmem} MHz"
        CLOCKS_LOCKED=1
    else
        echo "  clocks NOT locked (needs root / not supported) -- dispersion is reported, not assumed away"
        CLOCKS_LOCKED=0
    fi
}
unlock_clocks() { [ "${CLOCKS_LOCKED:-0}" = 1 ] && { nvidia-smi -rgc >/dev/null 2>&1; nvidia-smi -rmc >/dev/null 2>&1; echo "  clocks released"; }; }

# --------------------------------------------------------------------------------------------------
# Feasibility: what a grid size costs in device and host memory, before it is attempted
#
# Both models are exact allocations read off bench_27pt_precision and io.cu, not estimates. The host
# figure was checked against maxRSS at 192 cubed: predicted 5.38 GB, measured 5.375 GB. Getting this
# wrong on a rented instance costs the session, not a retry: a 448 cubed grid fills 80 GB of device
# memory but needs 68 GB of host RAM and several minutes of single-threaded generation per process.
# --------------------------------------------------------------------------------------------------
footprint_mb() {                  # $1 = N, $2 = "dev" | "host"
    awk -v N="$1" -v which="$2" 'BEGIN{
        n   = N*N*N;
        nnz = (3*N-2)^3;
        dev = 8*(n+1) + 4*nnz + 8*nnz + 4*nnz    \
            + 8*n + 8*(n + 2*N*N) + 8*n          \
            + 27*n*(8+4+2+2);
        # peak while the COO array and the CSR arrays are both live
        h1  = 16*nnz + 8*(n+1) + 4*nnz + 8*nnz + 4*n;
        # peak while CSR, the double SoA array and its three narrow copies are all live
        h2  = 12*nnz + 8*n + 27*n*8 + 27*n*(4+2+2);
        host = (h1 > h2) ? h1 : h2;
        printf "%d", ((which=="dev") ? dev : host)/1048576;
    }'
}

write_header() {                  # $1 = path, $2 = N, $3 = contrast ("" for constant)
    local rows=$(( $2 * $2 * $2 )) k=$(( 3 * $2 - 2 ))
    {
        printf '%%%%MatrixMarket matrix coordinate real general\n'
        printf '%% STENCIL_GRID_SIZE %d\n' "$2"
        [ -n "$3" ] && printf '%% STENCIL_CONTRAST %s\n' "$3"
        printf '%d %d %d\n' "$rows" "$rows" $(( k * k * k ))
    } > "$1"
}

# --------------------------------------------------------------------------------------------------
# B7
# --------------------------------------------------------------------------------------------------
# Traffic per row, from the byte model the benchmark itself reports: CSR double 240, SoA double 232,
# SoA float 124, SoA half 70. A ratio of times is only meaningful next to the ratio of bytes it
# would reach if traffic were the only cost, so every measured figure below is printed against its
# own traffic ceiling -- and the ceilings differ by base, which is the whole point of showing them.
run_b7() {
    local tag="$1"
    for N in $SIZES; do
        local devmb hostmb f
        devmb=$(footprint_mb "$N" dev); hostmb=$(footprint_mb "$N" host)
        printf '  N=%-4s device %5d MB / %5d MB    host %5d MB / %5d MB' "$N" "$devmb" "$VRAM_MB" "$hostmb" "$RAM_MB"
        if [ "$devmb" -gt $(( VRAM_MB * 92 / 100 )) ]; then echo "   SKIP (device)"; continue; fi
        if [ "$hostmb" -gt $(( RAM_MB  * 90 / 100 )) ]; then echo "   SKIP (host)";   continue; fi
        echo "   ok"
        f="matrix/stencil3d_27pt_${N}.mtx"
        write_header "$f" "$N" ""
        # Separate processes, not just more repetitions inside one. On the 4060 the median over
        # eleven repetitions is reproducible to a thousandth within a process and moves by up to a
        # tenth between processes, so repetitions alone measure the wrong dispersion.
        for p in $(seq 1 "$PROCS"); do
            printf '    process %d/%d ' "$p" "$PROCS"
            if ./bin/bench_27pt_precision "$f" --reps="$REPS" \
                   --csv="$OUT/b7_${tag}_N${N}_p${p}.csv" > "$OUT/b7_${tag}_N${N}_p${p}.log" 2>&1; then
                awk -F, '$5=="\"SoA float\""{printf "%.3f ms\n", $6}' "$OUT/b7_${tag}_N${N}_p${p}.csv"
            else
                echo "FAILED (see $OUT/b7_${tag}_N${N}_p${p}.log)"
            fi
        done
        b7_verdict "$tag" "$N"
    done
}

b7_verdict() {                    # $1 = tag, $2 = N
    local files; files=$(ls "$OUT"/b7_"$1"_N"$2"_p*.csv 2>/dev/null)
    [ -n "$files" ] || return 0
    awk -F, -v N="$2" '
        function strip(s){ gsub(/"/,"",s); return s }
        function med(a,k,   i,j,t,tmp,c){ c=0; for(i in a) tmp[++c]=a[i];
            for(i=2;i<=c;i++){t=tmp[i];j=i-1;while(j>0&&tmp[j]>t){tmp[j+1]=tmp[j];j--}tmp[j+1]=t}
            return (c%2)?tmp[int(c/2)+1]:(tmp[c/2]+tmp[c/2+1])/2 }
        FNR==1 { p++; next }
        { t[p,strip($5)] = $6+0; peak[p,strip($5)] = $9+0 }
        END{
            print "";
            printf "    --- B7 at %d^3, %d processes ---\n", N, p;
            printf "    %-34s %9s %9s %9s   %s\n", "ratio", "median", "min", "max", "traffic ceiling";
            split("", r1); split("", r2); split("", r3); split("", r4); split("", pk); split("", pkc); split("", pkd);
            for (i=1;i<=p;i++){
                r1[i] = t[i,"SoA double"] / t[i,"SoA float"];
                r2[i] = t[i,"CSR double"] / t[i,"SoA float"];
                r3[i] = t[i,"SoA double"] / t[i,"SoA half"];
                r4[i] = t[i,"SoA float"]  / t[i,"~probe accF32"];
                pk[i]  = peak[i,"SoA float"];
                pkc[i] = peak[i,"CSR double"];
                pkd[i] = peak[i,"SoA double"];
            }
            printf "    %-34s %9.3f %9.3f %9.3f   %.3f\n", "SoA f64 -> SoA f32  (precision)",  med(r1), mn(r1,p), mx(r1,p), 232.0/124.0;
            printf "    %-34s %9.3f %9.3f %9.3f   %s\n", "CSR f64 -> SoA f32  (production)", med(r2), mn(r2,p), mx(r2,p), "1.935 (see note)";
            printf "    %-34s %9.3f %9.3f %9.3f   %.3f\n", "SoA f64 -> SoA f16  (precision)",  med(r3), mn(r3,p), mx(r3,p), 232.0/70.0;
            printf "    %-34s %9.3f %9.3f %9.3f   %s\n",   "SoA f32 -> float accumulation",    med(r4), mn(r4,p), mx(r4,p), "1.000 (traffic identical)";
            printf "\n    %% of peak DRAM (median over processes):  CSR f64 %.1f%%   SoA f64 %.1f%%   SoA f32 %.1f%%\n", med(pkc), med(pkd), med(pk);
            print  "";
            print  "    Reading it: the last row is the discriminator. It moves no byte of traffic, so";
            print  "    anything above 1.0 is double-precision arithmetic emerging from behind the";
            print  "    memory floor. It sits well above 1 on a consumer part running FP64 at a";
            print  "    sixty-fourth of FP32. On a part that runs it at a half it should fall towards";
            print  "    1.0, and the two precision rows should rise towards their traffic ceilings.";
            print  "    Run this same script with --dry-run to get the local baseline it is read";
            print  "    against, rather than quoting a figure measured at another size.";
            if (med(r2) > 240.0/124.0) {
                print  "";
                print  "    NOTE: the production row exceeds its 1.935 figure. That is not a measurement";
                print  "    error and 1.935 is not a ceiling for it: a ratio of times can only be bounded";
                print  "    by a ratio of bytes when both kernels are at the memory wall, and the";
                print  "    row-major one is not. Its byte model counts USEFUL bytes; reading a row of";
                print  "    coefficients across lanes fetches 32-byte sectors to use 8 of them, so it";
                print  "    moves more than the model says. Compare the two % of peak below to see it.";
            }
            spread = (mx(r1,p) - mn(r1,p)) / med(r1);
            if (spread > 0.15) {
                print  "";
                printf "    WARNING: the precision ratio moves by %.0f%% across identical processes.\n", 100*spread;
                print  "    That dispersion is of the same order as the effect. Report the range, not a";
                print  "    single number, and raise --procs before drawing any conclusion from it.";
            }
        }
        function mn(a,c,  i,v){ v=a[1]; for(i=2;i<=c;i++) if(a[i]<v) v=a[i]; return v }
        function mx(a,c,  i,v){ v=a[1]; for(i=2;i<=c;i++) if(a[i]>v) v=a[i]; return v }
    ' $files
}

# --------------------------------------------------------------------------------------------------
# B6 and B1
# --------------------------------------------------------------------------------------------------
run_b6() {
    local tag="$1"
    log "B6 - SoA vs CSR at solver level, build=$tag"
    write_header "matrix/stencil3d_27pt_192.mtx" 192 ""
    for np in 1 2 4 8; do
        [ "$np" -le "$MAXG" ] || continue
        for mode in csr soa; do
            log "  np=$np spmv=$mode"
            mpirun --allow-run-as-root -np "$np" ./bin/cg_solver_mgpu_stencil_3d \
                matrix/stencil3d_27pt_192.mtx --stencil=27 --spmv="$mode" \
                > "$OUT/b6_${tag}_np${np}_${mode}.log" 2>&1 \
                && grep -iE "total|iteration|time|residual" "$OUT/b6_${tag}_np${np}_${mode}.log" | tail -6 \
                || echo "  FAILED (see $OUT/b6_${tag}_np${np}_${mode}.log)"
        done
    done
}

build() {                         # $1 = extra NVCCFLAGS ("" for the published configuration)
    make clean >/dev/null 2>&1
    local targets="bench_27pt_precision"
    [ "$ONLY" = b7 ] || targets="bench_27pt_precision cg_solver_mgpu_stencil_3d"
    # shellcheck disable=SC2086
    if [ -n "$1" ]; then
        make -j"$(nproc)" ARCH="$CC" NVCCFLAGS="$1" $targets >> "$OUT/build.log" 2>&1
    else
        make -j"$(nproc)" ARCH="$CC" $targets >> "$OUT/build.log" 2>&1
    fi
}

# --------------------------------------------------------------------------------------------------
# Session
# --------------------------------------------------------------------------------------------------
log "Session start"
echo "  GPU        : $GPU_NAME  (sm_${CC}, ${VRAM_MB} MB, ${NGPU_AVAIL} present)"
echo "  Host RAM   : ${RAM_MB} MB available"
echo "  Scope      : --only=$ONLY   sizes='$SIZES'   procs=$PROCS   reps=$REPS$([ "$DRYRUN" = 1 ] && echo '   [DRY RUN]')"
lock_clocks
nvidia-smi --query-gpu=index,clocks.sm,clocks.mem,temperature.gpu --format=csv > "$OUT/clocks_start.csv"
{ echo "gpu=$GPU_NAME"; echo "cc=$CC"; echo "vram_mb=$VRAM_MB"; echo "ram_mb=$RAM_MB";
  echo "sizes=$SIZES"; echo "procs=$PROCS"; echo "reps=$REPS"; echo "date=$(date -Is)"; } > "$OUT/session_info.txt"

log "Build at -O2, compiled offline for sm_${CC} (the configuration the published numbers use, plus an explicit target)"
build "" || { echo "build failed -- see $OUT/build.log"; tail -20 "$OUT/build.log"; unlock_clocks; exit 1; }
echo "  ok"

if [ "$ONLY" = b7 ] || [ "$ONLY" = all ]; then
    log "B7 - coefficient precision, single GPU"
    run_b7 o2
fi
if [ "$ONLY" = b6 ] || [ "$ONLY" = all ]; then
    run_b6 o2
fi
if [ "$ONLY" = b1 ] || [ "$ONLY" = all ]; then
    log "B1 - rebuild at -O3 and repeat"
    case "$GPU_NAME" in
        *A100*) ;;
        *) echo "  WARNING: the published -O2 figures are A100. On a $GPU_NAME this answers a"
           echo "           different question, and the two must not be merged into one table." ;;
    esac
    if build "-O3 --ptxas-options=-O3 --ptxas-options=-allow-expensive-optimizations=true -std=c++11"; then
        run_b7 o3
        [ "$ONLY" = all ] && run_b6 o3
    else
        echo "  -O3 build failed, skipping B1 (see $OUT/build.log)"
    fi
fi

nvidia-smi --query-gpu=index,clocks.sm,clocks.mem,temperature.gpu --format=csv > "$OUT/clocks_end.csv"
unlock_clocks

log "Done. Collecting."
tar czf results.tgz "$OUT"
echo
echo "  Bring back: results.tgz  ($(du -h results.tgz | cut -f1))"
echo "  Then, and only then, destroy the pod."
