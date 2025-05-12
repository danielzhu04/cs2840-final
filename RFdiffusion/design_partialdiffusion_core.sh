#!/bin/bash

# ./scripts/run_inference.py \
#     inference.output_prefix=data/RFdiffusion_output/esm3_abl1b_complete_partialdiffusion_core_t5 \
#     inference.input_pdb=data/ESM_original/esm3_abl1b_complete.pdb \
#     'contigmap.contigs=["1149-1149"]' \
#     inference.num_designs=1 \
#     diffuser.partial_T=5 \
#     'contigmap.provide_seq=[0-227,515-1149]' \
./scripts/run_inference.py \
    inference.output_prefix=data/RFdiffusion_output/esm3_abl1b_complete_partialdiffusion_core_t50_again \
    inference.input_pdb=data/ESM_original/esm3_abl1b_complete.pdb \
    'contigmap.contigs=["A1-228/287/A516-1149"]' \
    inference.num_designs=1 \
    diffuser.partial_T=50 
