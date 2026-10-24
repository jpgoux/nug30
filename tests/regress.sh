#!/bin/sh
# usage: regress.sh BIN [extra options]
# prints: instance optimum found nodes fw_iters seconds
BIN=$1; shift
DATA=${DATA:-../data/qaplib}
for x in nug12:578 nug14:1014 nug15:1150 nug16a:1610 nug16b:1240 nug17:1732; do
  i=${x%:*}; o=${x#*:}
  $BIN solve $DATA/$i.dat -strat C -ub $((o+1)) -nointprune "$@" | tail -1 | python3 -c '
import json,sys; d=json.loads(sys.stdin.read()); print(sys.argv[1], sys.argv[2], int(d["opt"]), d["nodes"], d["fw_iters"], d["wall_s"])' $i $o
done
