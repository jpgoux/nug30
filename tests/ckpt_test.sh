#!/bin/sh
# checkpoint test: solve nug17, interrupt twice, resume.
# 1. With children started from the centre (-warm 0), the total node count must be
#    the one of an uninterrupted run (same tree).
# 2. With the default start from the parent's point, the nodes resumed from the file
#    start from the centre (the parent's point is not saved), so the tree may differ
#    slightly: the optimum must be found and the node count stay within 1 %.
# usage: ckpt_test.sh BIN
BIN=$1; DATA=${DATA:-../data/qaplib}; T=$(mktemp -d)
O="solve $DATA/nug17.dat -strat C -ub 1733 -nointprune"
run3() {   # $1: extra options; prints the last checkpoint line
    timeout 3 $BIN $O $1 -ckpt $T/a -ckpt_s 1 > /dev/null
    timeout 2 $BIN $O $1 -resume $T/a -ckpt $T/b -ckpt_s 1 > /dev/null 2>&1
    $BIN $O $1 -resume $T/b -ckpt $T/c > /dev/null 2>&1
    head -1 $T/c
}
L0=$(run3 "-warm 0")
E0=$($BIN $O -warm 0 | tail -1 | sed 's/.*"nodes":\([0-9]*\).*/\1/')
L2=$(run3 "")
rm -rf $T
E2=$(grep '^nug17 ' $(dirname $0)/expected.txt | cut -d' ' -f4)
N2=$(echo "$L2" | sed 's/.* nodes \([0-9]*\) .*/\1/')
echo "$L0" | grep -q " ub 1732 nodes $E0 " || { echo "checkpoint: FAILED, exact resume ($L0, expected $E0 nodes)"; exit 1; }
echo "$L2" | grep -q " ub 1732 " && [ $((N2 * 100)) -le $((E2 * 101)) ] && [ $((N2 * 100)) -ge $((E2 * 99)) ] \
    || { echo "checkpoint: FAILED, resume with parent start ($L2, expected about $E2 nodes)"; exit 1; }
echo "checkpoint: OK (exact: $E0 nodes; parent start: $N2 nodes against $E2 uninterrupted)"
