
function get_distributed_rank() {
    if [ "$TF_TYPE" == "ps" ]; then
        echo 0
    else
        echo $(($TF_INDEX+1))
    fi
}

function echoerr() {
    echo "$@" 1>&2;
}

function num_gpu() {
    if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
        local num_gpu=`nvidia-smi | grep Default | wc -l`
    else
        # 优先CUDA_VISIBLE_DEVICES
        local num_gpu=`echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l`
    fi
    echoerr num_gpu: $num_gpu
    echo $num_gpu
}

function setup_train_env_deepspeed() {
    num_gpu=`num_gpu`
    if [ ! -z "$TF_TYPE" ]; then
        echoerr Distributed training...
        nnodes=`echo $TF_WHOSTS | tr ',' '\n' | wc -l`
        nnodes=`echo $(($nnodes + 1))`
        master_host=`echo $TF_PHOSTS | cut -f 1 -d ':'`
        master_port=`echo $TF_PHOSTS | cut -f 2 -d ':'`
        launch_prefix_cmd="python -m deepspeed.launcher.launch"
        echo $TF_PHOSTS | cut -d ':' -f 1 | \
           awk -v num_gpu=`num_gpu` '{print $1"\tslots="num_gpu}' >> /tmp/mpihost
        echo $TF_WHOSTS | sed 's/,/\n/g' | cut -d ':' -f 1 | \
           awk -v num_gpu=`num_gpu` '{print $1"\tslots="num_gpu}' >> /tmp/mpihost
        world_info=`python $(dirname "${BASH_SOURCE}")/hostfile2world_info.py -i /tmp/mpihost`
        launch_prefix_cmd+=" --world_info $world_info --node_rank `get_distributed_rank`"
        launch_prefix_cmd+=" --master_addr $master_host --master_port $master_port"
    elif [ $num_gpu -eq 1 ]; then
        nnodes=1
        launch_prefix_cmd="python "
    else
        nnodes=1
        local port=$(echo $((20000 + $RANDOM % 1000)))
        launch_prefix_cmd="python -m deepspeed.launcher.runner --num_nodes=$nnodes --master_port $port "
    fi
    world_size=`echo $(($nnodes * $num_gpu))`
    echoerr nnodes: $nnodes
    echoerr world_size: $world_size
    echoerr master_host: $master_host
    echoerr master_port: $master_port
    echoerr launch_prefix_cmd: $launch_prefix_cmd
    export NCCL_IB_DISABLE=1
    export WORLD_SIZE=$world_size
    export NUM_NODE=$nnodes
    #export NCCL_SOCKET_IFNAME=eth0
}


function setup_train_env_torchrun() {
    num_gpu=`num_gpu`
    if [ ! -z "$TF_TYPE" ]; then
        echoerr Distributed training...
        nnodes=`echo $TF_WHOSTS | tr ',' '\n' | wc -l`
        nnodes=`echo $(($nnodes + 1))`
        master_host=`echo $TF_PHOSTS | cut -f 1 -d ':'`
        master_port=`echo $TF_PHOSTS | cut -f 2 -d ':'`
        launch_prefix_cmd="torchrun "

        launch_prefix_cmd+=" --nproc_per_node=$num_gpu"
        launch_prefix_cmd+=" --nnodes $nnodes --node_rank `get_distributed_rank`"
        launch_prefix_cmd+=" --master_addr $master_host --master_port $master_port "
    elif [ $num_gpu -eq 1 ]; then
        nnodes=1
        launch_prefix_cmd="python "
    else
        nnodes=1
        local port=$(echo $((20000 + $RANDOM % 1000)))
        launch_prefix_cmd="torchrun --nproc_per_node=$num_gpu --master_port=$port "
    fi
    world_size=`echo $(($nnodes * $num_gpu))`
    echoerr nnodes: $nnodes
    echoerr world_size: $world_size
    echoerr master_host: $master_host
    echoerr master_port: $master_port
    echoerr launch_prefix_cmd: $launch_prefix_cmd
    export NCCL_IB_DISABLE=1
    export WORLD_SIZE=$world_size
    export NUM_NODE=$nnodes
    export launch_prefix_cmd_torch=$launch_prefix_cmd
    #export NCCL_SOCKET_IFNAME=eth0
}


function setup_train_env_accelerate() {
    num_gpu=`num_gpu`
    if [ ! -z "$TF_TYPE" ]; then
        echoerr Distributed training...
        nnodes=`echo $TF_WHOSTS | tr ',' '\n' | wc -l`
        nnodes=`echo $(($nnodes + 1))`
        master_host=`echo $TF_PHOSTS | cut -f 1 -d ':'`
        master_port=`echo $TF_PHOSTS | cut -f 2 -d ':'`
        launch_prefix_cmd="accelerate launch "
        echo $TF_PHOSTS | cut -d ':' -f 1 | \
           awk -v num_gpu=`num_gpu` '{print $1"\tslots="num_gpu}' >> /tmp/mpihost
        echo $TF_WHOSTS | sed 's/,/\n/g' | cut -d ':' -f 1 | \
           awk -v num_gpu=`num_gpu` '{print $1"\tslots="num_gpu}' >> /tmp/mpihost
        launch_prefix_cmd+=" --main_process_ip $master_host --main_process_port $master_port"
        world_size=`echo $(($nnodes * $num_gpu))`
        launch_prefix_cmd+=" --machine_rank `get_distributed_rank` --num_processes $world_size --num_machines $nnodes"
    elif [ $num_gpu -eq 1 ]; then
        nnodes=1
        launch_prefix_cmd="python "
    else
        nnodes=1
        local port=$(echo $((20000 + $RANDOM % 1000)))
        launch_prefix_cmd="accelerate launch "
    fi
    world_size=`echo $(($nnodes * $num_gpu))`
    echoerr nnodes: $nnodes
    echoerr world_size: $world_size
    echoerr master_host: $master_host
    echoerr master_port: $master_port
    echoerr launch_prefix_cmd: $launch_prefix_cmd
    export MASTER_ADDR=$master_host
    export MASTER_PORT=$(echo $($master_port + 1))
    export NCCL_IB_DISABLE=1
    export WORLD_SIZE=$world_size
    export NUM_NODE=$nnodes
    export launch_prefix_cmd_accelerate=$launch_prefix_cmd
    #export NCCL_SOCKET_IFNAME=eth0
}

function setup_train_env() {
    setup_train_env_torchrun
    setup_train_env_accelerate
    setup_train_env_deepspeed
}

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

function singleton_setup() {
    export GPT_HOME=/tmp/
    export LD_LIBRARY_PATH=$GPT_HOME/python310/lib:$LD_LIBRARY_PATH
    export LIBRARY_PATH=$GPT_HOME/python310/lib:$LIBRARY_PATH
    export PATH=$GPT_HOME/python310/bin:$PATH
    #export LD_LIBRARY_PATH=$GPT_HOME/python310/lib:$LD_LIBRARY_PATH
    #export LIBRARY_PATH=$GPT_HOME/python310/lib:$LIBRARY_PATH
    #export PATH=$GPT_HOME/python310/bin:$PATH

    __config_cuda_path $CCX_HOME/tools/opt/libcudnn
    __config_cuda_path $CCX_HOME/tools/opt/cuda-11.7
    __config_opt_path $CCX_HOME/tools/opt/nccl_cuda11.7
    __config_opt_path $CCX_HOME/tools/opt/cuda-11.7

    ps -ef | grep trainer | grep -v grep | tee /dev/stderr | awk '{print $2}' | xargs kill -9
    #ps -ef | grep main.py | grep -v grep | tee /dev/stderr | awk '{print $2}' | xargs kill -9
    rm -rf /tmp/data_files/
    #rm -rf ~/.cache/huggingface/datasets

    setup_train_env
    if [ "$IS_SETUP" == "1" ]; then
        echo 'already setup.'
        return
    fi
    export IS_SETUP="1"


    python -m pip uninstall -y deepspeed
    #python -m pip install deepspeed==0.9.5
    python -m pip install -e ~/DeepSpeed
    #--no-dependencies
    #python -m pip uninstall -y transformers
    python -m pip install -e ~/transformers --no-dependencies
    #python -m pip uninstall -y alpaca_farm
    python -m pip install -e ~/alpaca_farm --no-dependencies
    #python -m pip uninstall -y accelerate
    python -m pip install -e ~/accelerate/ --no-dependencies
    #python -m pip install fairscale --no-dependencies
    python -m pip install ~/wheel/torch-2.1.0.dev20230728+cu118-cp310-cp310-linux_x86_64.whl --no-dependencies
}
singleton_setup
