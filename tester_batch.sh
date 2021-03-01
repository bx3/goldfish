#!/bin/bash
trap kill_batch INT

function kill_batch() {
	echo "kill all pid"
	for pid in $pids; do
		kill $pid
	done
	exit 0
}

pids=""
exp_name="linearAlgoInit"
for num_mask in 0 5 10 15; do
	for noise_std in 0 5 10 15; do
		mkdir -p analysis/tester_exp/${exp_name} || exit 1
		logname="analysis/tester_exp/${exp_name}/node20-region5-noise${noise_std}-1D-linear-append1msg-bandit-${num_mask}mask-algo-"
		echo "running ${logname}"
		./testbed.py mf-online 1 node=20 region=5 new_msgs=1 H_method=1D-linear add_method=append mask_method=bandit init_method=algo name=${exp_name} num_mask=${num_mask} noise=${noise_std} max_iter=50 > ${logname} &
		pid="$!"
		pids="$pids $pid"
	done
done

echo $pids

for pid in $pids; do
	wait $pid
done

echo "Done."


