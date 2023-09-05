python -m pip install tensorboardx

for logdir in $@; do
    echo $logdir
    set -x
    rm -rf $logdir/runs/
    sleep 20
    if [ -f $logdir/training_0.log ]; then
        fn=$logdir/training_0.log
    else
        fn=$logdir/training_1.log
    fi
    cat $fn | \
        grep -E 'correct|reward|loss|step|obj|ppo|return|val' | sed 's/|/\n/g' | python ./utils/kv2tensorboard.py -o $logdir/runs/
    if [ -d $logdir/rollouts/ ]; then
        rm -rf $logdir/rollouts/runs/
        sleep 20
        python ./test_rollout.py -i $logdir/rollouts/ | python ./utils/kv2tensorboard.py -o $logdir/rollouts/runs/
    fi
done

ps -ef | grep tensorboard | grep -v grep | tee /dev/stderr | awk '{print $2}' | xargs kill -9
nohup tensorboard --logdir=. --port 8081 --host 0.0.0.0 &

