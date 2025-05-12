mv # .bashrc

# Source global definitions
if [ -f /etc/bashrc ]; then
	. /etc/bashrc
fi

# User specific aliases and functions

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
# __conda_setup="$('/gpfs/runtime/opt/anaconda/2022.05/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
# if [ $? -eq 0 ]; then
#     eval "$__conda_setup"
# else
#     if [ -f "/gpfs/runtime/opt/anaconda/2022.05/etc/profile.d/conda.sh" ]; then
#         . "/gpfs/runtime/opt/anaconda/2022.05/etc/profile.d/conda.sh"
#     else
#         export PATH="/gpfs/runtime/opt/anaconda/2022.05/bin:$PATH"
#     fi
# fi
# unset __conda_setup
# <<< conda initialize <<<
# alias cs1430_env='module load anaconda/2022.05 python/3.11.0 openssl/3.0.0 cudnn/8.6.0 cuda/11.7.1 gcc/10.2 && conda activate cs1430_env'
# export LD_LIBRARY_PATH=/users/wli115/anaconda/cs1430_env/lib:$LD_LIBRARY_PATH #Let's see if this bricks it lol
# alias tf_2.9_env='module load anaconda/2022.05 python/3.9.0 openssl/3.0.0 cudnn/8.6.0 cuda/11.7.1 gcc/10.2 && conda activate tf_2.9'
# export LD_LIBRARY_PATH=/users/wli115/scratch/LittleLabelLearners/tf.venv/lib64/python3.x/site-packages/nvidia/cublas/lib:/users/wli115/scratch/LittleLabelLearners/tf.venv/lib64/python3.x/site-packages/nvidia/cudnn/lib

# export CUDNN_PATH="/users/wli115/scratch/LittleLabelLearners/tf.venv/lib64/python3.x/site-packages/nvidia/cudnn"
# export LD_LIBRARY_PATH="$CUDNN_PATH/lib":"/users/wli115/scratch/LittleLabelLearners/tf.venv/lib64"
# export PATH="$PATH":"/users/wli115/scratch/LittleLabelLearners/tf.venv/bin"

# Shortcuts for running different interactive instances
function int_gpu() {
    interact -q gpu -g "${1:-1}" -f "${2:-ampere}" -m "${3:-192}"g -n "${4:-1}" -t 48:00:00;
    # interact -q gpu -g 1 -f "${1:-ampere}" -m "${2:-192}"g -n "${3:-1}" -t 20:00:00;
    # interact -q gpu -g 1 -f "${1:-ampere}" -m "${2:-192}"g -t 20:00:00;
}

function int_gpu_multi() {
    interact -q gpu -g "${1:-2}" -f "${2:-ampere}" -m "${3:-192}"g -n "${4:-1}" -t 24:00:00;
    # interact -q gpu -g 1 -f "${1:-ampere}" -m "${2:-192}"g -n "${3:-1}" -t 20:00:00;
    # interact -q gpu -g 1 -f "${1:-ampere}" -m "${2:-192}"g -t 20:00:00;
}

function int_3090_gcondo() {
    # interact -q 3090-gcondo -g "${1:-8}" -m "${2:-192}"g -n "${3:-1}" -t 12:00:00;
    interact -q 3090-gcondo -f "${1:-ampere}" -g "${2:-8}" -m "${3:-192}"g -n "${4:-1}" -t 12:00:00;
    # interact -q gpu -g 1 -f "${1:-ampere}" -m "${2:-192}"g -n "${3:-1}" -t 20:00:00;
    # interact -q gpu -g 1 -f "${1:-ampere}" -m "${2:-192}"g -t 20:00:00;
}

function int_rsingh47_gcondo() {
    interact -q rsingh47-gcondo  -g "${1:-4}" -m "${2:-192}"g -n "${3:-1}" -t 12:00:00;
}

# alias int_gpu='interact -q gpu -g 1 -f a5000 -m 192g -n 12 -t 20:00:00'
alias int_mem='interact -q bigmem -m 384g -n 12 -t 20:00:00'

function notebook_setup() {
    echo "setting up jupyter notebook";
    module load cuda cudnn;
    ipnport=$(shuf -i8000-9999 -n1); 
    echo ipnport=$ipnport;
    ipnip=$(hostname -i);
    echo ipnpip=$ipnip;
    echo ssh -N -L $ipnport:$ipnip:$ipnport wli115@ssh.ccv.brown.edu
    echo localhost:$ipnport

    jupyter-notebook --no-browser --port=$ipnport --ip=$ipnip;
}

export SCREENDIR=$HOME/.screen


export APPTAINER_CACHEDIR=/tmp
export APPTAINER_TMPDIR=/tmp

export APPTAINER_BINDPATH="/oscar/home/$USER,/oscar/scratch/$USER,/oscar/data"
alias run_apptainer="apptainer run --nv /oscar/runtime/software/external/ngc-containers/tensorflow.d/x86_64.d/tensorflow-24.03-tf2-py3.simg"
