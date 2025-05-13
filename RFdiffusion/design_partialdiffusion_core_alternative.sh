#!/bin/bash
./scripts/run_inference.py \
    inference.output_prefix=data/RFdiffusion_output/t2/alternative/esm3_abl1b_complete_partialdiffusion_core \
    inference.input_pdb=data/ESM_original/esm3_abl1b_complete.pdb \
    'contigmap.contigs=["1149-1149"]' \
    diffuser.partial_T=2 \
    inference.num_designs=1 \
    'contigmap.provide_seq=[0-228,515-1148]'
