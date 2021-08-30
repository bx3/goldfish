#!/bin/bash
trap kill_procs SIGINT

function kill_procs() {
    echo "STOP"
    for pid in $pids; do
        echo "kill process $pid"
        kill $pid
    done
}

function run_local() {
	if [ $# -ne 6 ]; then
		echo "./run.sh run-local num-node<int> num-pub<int> num-out<int> num-in<int> num-msg<int> num-epoch<int>"
		echo "./run.sh run-local 10 2 4 10 10"
		exit 1
	fi
	num_node=$1
	num_pub=$2
	num_out=$3
	num_in=$4
	num_msg=$5
	num_epoch=$6
    rm -rf stores
    mkdir stores

	gen_local_topo ${num_node} ${num_pub} 30000

	pids=""
	for (( i=0; i<${num_node}; i++ )); do
		./main run "configs/node$i.json" ${num_out} ${num_in} ${num_msg} ${num_epoch} &
		pids="$pids $!" 
	done

	for pid in $pids; do
		wait $pid
	done
}

function gen_local_topo() {
	if [ $# -ne 3 ]; then
		echo "./run.sh gen-local-topo num-node<int> port<int>"
		exit 1
	fi
	num_node=$1
	num_pub=$2
	port=$3
	./script/gen-local-topo.py ${num_node} ${num_pub} ${port}
}

case "$1" in
  help)
		echo "./run.sh subcommand[run-local/gen-local-topo]" ;;
	run-local)
		run_local ${@:2} ;;
	gen-local-topo)
		gen_local_topo ${@:2} ;;
	*)
		tput setaf 1
		echo "Unknown subcommand $1"
		echo "./run.sh help"
		tput sgr 0 ;;
esac
