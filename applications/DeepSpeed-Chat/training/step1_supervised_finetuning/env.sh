
function __config_cuda_path() {
    local cuda_path=$1
    export CUDA_PATH=$cuda_path
    export LD_LIBRARY_PATH=$CUDA_PATH/lib64/:$CUDA_PATH/lib64/stubs/:$LD_LIBRARY_PATH
}
function __config_opt_path() {
    local __opt_module_home=$1
    export PATH=$__opt_module_home/bin/:$PATH

    export LD_LIBRARY_PATH=$__opt_module_home/lib/:$LD_LIBRARY_PATH
    export LIBRARY_PATH=$__opt_module_home/lib/:$LIBRARY_PATH
}
export GPT_HOME=/tmp/
export LD_LIBRARY_PATH=$GPT_HOME/python39/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=$GPT_HOME/python39/lib:$LIBRARY_PATH
export PATH=$GPT_HOME/python39/bin:$PATH

__config_cuda_path $CCX_HOME/tools/opt/libcudnn
__config_cuda_path $CCX_HOME/tools/opt/cuda-11.3
__config_opt_path $CCX_HOME/tools/opt/nccl_cuda11.3
__config_opt_path $CCX_HOME/tools/opt/cuda-11.3
