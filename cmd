#!/bin/bash
trap kill_batch INT

if [ $1 = "help" ]; then
	# echo "./cmd subcommand[run-mf] name[string] seed[int] out_lim[int] num_region[int] useNodeHash[y/n] roundList[intList]"
	python testbed.py help 1
	exit 0
fi

if [ "$#" -le 3 ]; then
  echo "Error. Argument invalid"
	exit 0
fi

function kill_batch() {
	exit 0
}

subcommand=$1
name=$2
seed=$3
out_lim=$4
num_region=$5
use_node_hash=$6

record_round="${@:7}"

dirname="${name}_seed${seed}"
dirpath="analysis/$dirname"
mkdir $dirpath
cp *.py $dirpath

# run experiment
if [ ${subcommand} = 'run' ]; then 
	python testbed.py run ${seed} ${dirpath} ${out_lim} ${num_region} ${use_node_hash} ${record_round}
else
	python testbed.py complete_graph ${seed} ${dirpath} ${out_lim} ${use_node_hash} ${record_round}
fi
retval=$?
if [ "$retval" -ne 0 ]; then
	echo "simulation bug. Exit"
	exit 1
fi	

# Calculate it
cd analysis 
cal_cmd="./CalculateDelay_batch.py subset ${use_node_hash} ${dirname} ${record_round}"
echo ${cal_cmd}
${cal_cmd}

# Plot it 
plot_cmd="./plot_single.py ${dirname} ${seed} ${record_round}"
echo ${plot_cmd}
${plot_cmd}

