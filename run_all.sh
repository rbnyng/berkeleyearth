#!/usr/bin/env bash
# Reproduce the certificate, the numerical studies and the figures.
# SMOKE=1 ./run_all.sh runs reduced-size versions of the floating-point experiments.
set -euo pipefail
cd "$(dirname "$0")"
export CERT_OUT="${CERT_OUT:-$PWD/arb_certificate/results}"

echo "== 1. Rigorous certificate (ball arithmetic) =="
python3 arb_certificate/bfej.py 700          # S^2 finite scan, n <= 700        -> BFEJ_700.json
python3 arb_certificate/final.py             # ||R^(20)||_1 and N0 (Theorem 2)  -> final.json
python3 arb_certificate/certificate.py       # S^3 scan, assembles everything   -> CERTIFICATE.json
python3 arb_certificate/validate.py          # closed form, main term, residual vs bound

echo "== 2. Floating-point cross-checks and experiments =="
python3 verify_all.py
python3 exp_invisibility.py                  # floors.npy  (Fig. 3a)
python3 exp_phase.py                         # phase.npy   (Fig. 3b)
python3 exp_matern.py                        # matern.npy  (Fig. 4, Table 3)
python3 exp_networks.py                      # Table 2 (S^3, random-design and SO(3) rows); Sec. 6.2 (i)

echo "== 3. Figures =="
python3 make_figures.py
