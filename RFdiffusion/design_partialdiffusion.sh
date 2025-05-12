#!/bin/bash

./scripts/run_inference.py \
    inference.output_prefix=output_pdbs/esm3_abl1b_complete_partialdiffusion_t1 \
    inference.input_pdb=input_pdbs/esm3_abl1b_complete.pdb \
    'contigmap.contigs=[1149-1149]' \
    inference.num_designs=1 \
    diffuser.partial_T=1
