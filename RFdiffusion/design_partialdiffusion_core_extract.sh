#!/usr/bin/env bash

set -euo pipefail                 # fail fast, catch unset vars & pipe errors

# ───────────────────────── arguments / defaults ────────────────────────────
PARTIAL_T=${1:-1}                 # 1st CLI arg (default 1 if none given)

# (optional) second arg → number of designs; default 1
NUM_DESIGNS=${2:-1}

# ───────────────────────── derived output prefix ───────────────────────────
OUTPUT_PREFIX="../data/RFdiffusion_output/esm3_abl1b_complete_partialdiffusion_core_t${PARTIAL_T}_combined_again_lmao"

echo "→ partial_T      = ${PARTIAL_T}"
echo "→ num_designs    = ${NUM_DESIGNS}"
echo "→ output_prefix  = ${OUTPUT_PREFIX}"

# ───────────────────────── run RFdiffusion ─────────────────────────────────
./scripts/run_inference.py \
  inference.output_prefix="${OUTPUT_PREFIX}" \
  inference.input_pdb=data/ESM_original/esm3_abl1b_complete.pdb \
  'contigmap.contigs=["A1-228/287/A516-1149"]' \
  inference.num_designs="${NUM_DESIGNS}" \
  diffuser.partial_T="${PARTIAL_T}"

# ───────────────────────── trim the resulting PDB ──────────────────────────
python - <<PYTHON
from pathlib import Path
from Bio.PDB import PDBParser, PDBIO, Select

prefix    = "${OUTPUT_PREFIX}"
src_path  = Path(f"{prefix}_0.pdb")
trim_path = src_path.with_name(src_path.stem + "_229-515.pdb")

parser    = PDBParser(QUIET=True)
structure = parser.get_structure("rf", src_path)

class KeepRange(Select):
    def accept_residue(self, residue):
        hetflag, resseq, icode = residue.id
        return hetflag == " " and 229 <= resseq <= 515   # chain A only

io = PDBIO()
io.set_structure(structure)
io.save(str(trim_path), select=KeepRange())
print(f"✅  Wrote trimmed PDB → {trim_path}")
PYTHON
