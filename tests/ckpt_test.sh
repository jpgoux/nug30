#!/bin/sh
# checkpoint test: solve nug17, interrupt twice, resume; the total node count must be
# the one of an uninterrupted run (same tree). usage: ckpt_test.sh BIN
BIN=$1; DATA=${DATA:-../data/qaplib}; T=$(mktemp -d)
O="solve $DATA/nug17.dat -strat C -ub 1733 -nointprune"
timeout 10 $BIN $O -ckpt $T/a -ckpt_s 3 > /dev/null
timeout 8 $BIN $O -resume $T/a -ckpt $T/b -ckpt_s 3 > /dev/null 2>&1
$BIN $O -resume $T/b -ckpt $T/c > /dev/null 2>&1
L=$(head -1 $T/c); rm -rf $T
echo "$L" | grep -q ' ub 1732 nodes 58818 ' && echo "checkpoint: OK ($L)" || { echo "checkpoint: FAILED ($L)"; exit 1; }
