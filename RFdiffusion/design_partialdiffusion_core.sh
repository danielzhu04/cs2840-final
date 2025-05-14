#!/bin/bash
./scripts/run_inference.py \
    inference.output_prefix=data/RFdiffusion_output/t1/esm3_abl1b_complete_partialdiffusion_core \
    inference.input_pdb=data/ESM_original/esm3_abl1b_complete.pdb \
    'contigmap.contigs=["A1-228/287/A516-1149"]' \
    inference.num_designs=100 \
    diffuser.partial_T=1
