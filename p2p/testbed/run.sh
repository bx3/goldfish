#!/bin/bash
trap kill_procs SIGINT

function kill_procs() {
    echo "STOP"
    for pid in $pids; do
        echo "kill process $pid"
        kill $pid
    done
}

experiment_port=30000

function run_local() {
	if [ $# -ne 9 ]; then
		echo "./run.sh run-local num-node<int> num-pub<int> num-adapt<int> num-out<int> num-in<int> num-rand<int> num-msg<int> num-epoch<int> rate<int>"
		echo "./run.sh run-local 10 1 1 3 6 1 10 3 1"
		exit 1
	fi
	num_node=$1
	num_pub=$2
    num_adapt=$3
	num_out=$4
	num_in=$5
    num_rand=$6
	num_msg=$7
	num_epoch=$8
    rate=$9
    rm -rf stores
    mkdir stores
    rm -rf configs
    mkdir configs

	gen_local_topo ${num_node} ${num_pub} ${num_adapt} ${experiment_port}

	pids=""
	for (( i=0; i<${num_node}; i++ )); do
		./main run "configs/node$i.json" ${num_out} ${num_in} ${num_rand} ${num_msg} ${num_epoch} $rate &
		pids="$pids $!" 
	done

	for pid in $pids; do
		wait $pid
	done
}

function gen_local_topo() {
	if [ $# -ne 4 ]; then
		echo "./run.sh gen-local-topo num-node<int> num-adapt<int> port<int>"
		exit 1
	fi
	num_node=$1
	num_pub=$2
    num_adapt=$3
	port=$4
	./script/gen-local-topo.py ${num_node} ${num_pub} ${num_adapt} ${port}
}

function plot_local() {
    if [ $# -ne 3 ]; then
		echo "./run.sh gen-local-topo num-node<int> num-epoch<int> data-path<str>"
		exit 1
	fi
    num_node=$1
	num_epoch=$2
	prefix=$3
    python ./script/plot_p2p.py ${num_node} ${num_epoch} ${prefix}
}

case "$1" in
  help)
		echo "./run.sh subcommand[run-local/gen-local-topo]" ;;
	run-local)
		run_local ${@:2} ;;
	gen-local-topo)
		gen_local_topo ${@:2} ;;
    plot-local)
        plot_local ${@:2} ;;
	*)
		tput setaf 1
		echo "Unknown subcommand $1"
		echo "./run.sh help"
		tput sgr 0 ;;
esac
