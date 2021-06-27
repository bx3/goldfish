#!/bin/bash
trap kill_batch INT

if [ $1 = "help" ]; then
	echo "./cmd subcommand[run-mf/run-2hop/complete-graph] seed[int] name[str]"
    echo "Plot json file. ./script/plot_topo.py inputs/dc_nodes10.json dc_10"
	python testbed.py help
	exit 0
fi

if [ "$#" -lt 1 ]; then
  echo "Error. Argument invalid"
	exit 0
fi

function kill_batch() {
	exit 0
}

subcommand=$1


# run experiment


if [ ${subcommand} = 'run' ]; then 
    seed=$2
    name=$3
    out_lim=$4
    num_region=$5
    num_new=$6
    use_node_hash=$7
    input_json=$8
    end_round=$9
    record_round=$(seq 0 $end_round)

    dirname="${name}-seed${seed}"
    dirpath="analysis/$dirname"
    mkdir $dirpath
    cp *.py $dirpath
    python testbed.py ${subcommand} ${seed} ${dirpath} ${out_lim} ${num_region} ${num_new} ${use_node_hash} ${input_json} ${record_round}
elif [ ${subcommand} = 'run-mc' ]; then
    seed=$2
    name=$3
    out_lim=$4
    num_new=$5
    topo=$6
    end_round=$7
    record_round=$(seq 0 $end_round)

    dirname="${name}-${out_lim}out-${num_new}new-seed${seed}"
    dirpath="output/$dirname"
    rm -rf $dirpath
    mkdir -p $dirpath
    mkdir -p $dirpath/src
    cp *.py $dirpath/src
    cmd="python testbed.py ${subcommand} ${seed} ${dirpath} ${out_lim} ${num_new} ${topo} ${record_round}"
    echo $cmd > $dirpath/experiment.log
    $cmd
    
    plot_cmd="python script/plot_single.py ${dirpath} $seed $topo 0.9 node ${record_round}"
    echo ${plot_cmd} >> $dirpath/experiment.log
    ${plot_cmd}
    echo "Run ->"
    echo "scp turing:/home/bowen/system-network/perigee-bandit-ml/${dirpath}/latest.png ."
    
elif [ ${subcommand} = 'gen-rand-topo' ]; then
    num_node=$2
    num_pub=$3
    name=$4
    ./script/gen-rand.py ${num_node} ${num_pub} > ./topo/${name}.json
    ./script/plot_topo.py ./topo/${name}.json ./topo/${name}
    echo "finish plot"

    echo "Run ->"
    echo "scp turing:/home/bowen/system-network/perigee-bandit-ml/topo/${name}.png . && open ${name}.png"
    exit 0
elif [ ${subcommand} = 'gen-datacenter' ]; then
    if [ $# -ne 10 ]; then
        echo "Require arguents"
        echo "./run.sh gen-datacenter use_single<y/n> num_center<int> num_node_per_center<int> num_pub_per_center<int> distance_among_center<int> center_std<int> name<str> proc_delay_meaen<int> proc_delay_std<int>"
        exit 0
    fi

    use_single=$2 
    num_center=$3 
    num_node_per_center=$4 
    num_pub_per_center=$5 
    dis_among_center=$6 
    center_std=$7
    name=$8
    proc_delay_mean=$9
    proc_delay_std=${10}
    ./script/gen-datacenter.py ${use_single} ${num_center} ${num_node_per_center} ${num_pub_per_center} ${dis_among_center} ${center_std} ${proc_delay_mean} ${proc_delay_std} > ./topo/${name}.json
    ./script/plot_topo.py ./topo/${name}.json ./topo/${name}
    echo "Run ->"
    echo "scp turing:/home/bowen/system-network/perigee-bandit-ml/topo/${name}.png . && open ${name}.png"
    exit 0
else
    echo "Unknown Subcommand ${subcommand}"
	exit 1
fi
retval=$?
if [ "$retval" -ne 0 ]; then
	echo "simulation bug. Exit"
	exit 1
fi	



# Calculate it
#cd analysis 
#cal_cmd="./CalculateDelay_batch.py subset n ${dirname} ${record_round}"
#echo ${cal_cmd}
#${cal_cmd}

## Plot it 
#plot_cmd="./plot_single.py ${dirname} ${seed} ${record_round}"
#echo ${plot_cmd}
#${plot_cmd}



