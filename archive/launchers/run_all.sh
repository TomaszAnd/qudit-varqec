#!/bin/bash
# Run all d=2 training scripts sequentially (~25 min total)
set -e
cd "$(dirname "$0")/.."

echo "=== Dephasing d=2 (1 layer, ~3 min) ==="
python3 scripts/train_dephasing_d2.py "$@"

echo ""
echo "=== Depolarizing d=2 (2 layers, ~10 min) ==="
python3 scripts/train_depolarizing_d2.py "$@"

echo ""
echo "=== Correlated-simplified d=2 (2 layers, ~10 min) ==="
python3 scripts/train_correlated_simplified_d2.py "$@"

echo ""
echo "=== All d=2 training complete ==="
echo "Note: dephasing d=3 not included (takes ~5 hrs). Run separately:"
echo "  python3 scripts/train_dephasing_d3.py"
