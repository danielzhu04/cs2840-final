#!/bin/bash

./scripts/run_inference.py \
    inference.output_prefix=data/RFdiffusion_output/esm3_abl1b_complete_partialdiffusion_t2 \
    inference.input_pdb=data/ESM_original/esm3_abl1b_complete.pdb \
    'contigmap.contigs=["A1-228/287-287/B516-1149"]' \
    inference.num_designs=1 \
    diffuser.partial_T=2
