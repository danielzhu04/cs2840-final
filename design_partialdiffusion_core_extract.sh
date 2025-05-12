#!/bin/bash

set -e            # stop on first error
set -o pipefail   # safer piping


OUTPUT_PREFIX="data/RFdiffusion_output/esm3_abl1b_complete_partialdiffusion_core_t1_combined_again"

./scripts/run_inference.py \
      inference.output_prefix="${OUTPUT_PREFIX}" \
      inference.input_pdb=data/ESM_original/esm3_abl1b_complete.pdb \
      'contigmap.contigs=[A1-228/287/A516-1149]' \
      inference.num_designs=1 \
      diffuser.partial_T=1

python - <<PYTHON
from pathlib import Path
from Bio.PDB import PDBParser, PDBIO, Select

# --- input / output paths ------------------------------------
prefix    = "${OUTPUT_PREFIX}"
src_path  = Path(f"{prefix}_0.pdb")          # RFdiffusion puts _0, _1, ...
trim_path = src_path.with_name(src_path.stem + "_229-515.pdb")

# --- load structure ------------------------------------------
parser    = PDBParser(QUIET=True)
structure = parser.get_structure("rf", src_path)

# --- selector -------------------------------------------------
class KeepRange(Select):
    def accept_residue(self, residue):
        hetflag, resseq, icode = residue.id
        return hetflag == " " and 229 <= resseq <= 515   # chain A only

# --- write ----------------------------------------------------
io = PDBIO()
io.set_structure(structure)
io.save(str(trim_path), select=KeepRange())
print(f"Wrote trimmed PDB -> {trim_path}")
PYTHON
