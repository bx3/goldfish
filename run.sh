#!/bin/bash
trap kill_batch INT

function kill_batch() {
	exit 0
}

function run {
    seed=$2
    name=$3
    out_lim=$4
    num_region=$5
    num_new=$6
    use_node_hash=$7
    input_json=$8
    end_epoch=$9
    record_round=$(seq 0 $end_epoch)

    dirname="${name}-seed${seed}"
    dirpath="analysis/$dirname"
    mkdir $dirpath
    cp *.py $dirpath
    python testbed.py ${subcommand} ${seed} ${dirpath} ${out_lim} ${num_region} ${num_new} ${use_node_hash} ${input_json} ${record_round}
}

function run_command {
    cmd=$1
    $cmd
    retval=$?
    if [ "$retval" -ne 0 ]; then
        echo "simulation bug. Exit"
        exit 1
    fi
}

function run_mc {
    if [ $# -ne 8 ]; then
        echo "./run.sh run-mc seed<int> out_lim<int> num_new<int> topo<str> end_epoch<int> name<str> use_logger<y/n>"
        exit 1
    fi

    seed=$2
    out_lim=$3
    num_new=$4
    topo=$5
    end_epoch=$6
    name=$7
    use_logger=$8
    record_round=$(seq 0 $end_epoch)

    dirname="${name}-${out_lim}out-${num_new}new-seed${seed}-${end_epoch}epoch"
    dirpath="output/$dirname"
    rm -rf $dirpath
    mkdir -p $dirpath
    mkdir -p $dirpath/src
    cp *.py $dirpath/src

    echo $(date '+%Y-%m-%d %H:%M:%S')  > $dirpath/experiment.log
    echo "./run.sh $@" >> $dirpath/experiment.log

    cmd="python testbed.py ${subcommand} ${seed} ${dirpath} ${out_lim} ${num_new} ${topo} ${use_logger} ${record_round}"
    echo $cmd >> $dirpath/experiment.log

    run_command "$cmd"
    
    plot_cmd="python script/plot_single.py ${dirpath} $topo 90 node ${record_round}"

    echo $(date '+%Y-%m-%d %H:%M:%S')  >> $dirpath/experiment.log
    echo ${plot_cmd} >> $dirpath/experiment.log

    run_command "${plot_cmd}"
    echo "Run ->"
    echo "scp turing:/home/bowen/system-network/perigee-bandit-ml/${dirpath}/latest.png ."
}


function run_2hop {
    if [ $# -ne 10 ]; then
        echo "./run.sh run-2hop seed<int> out_lim<int> num_msg_per_topo<int> topo<str> end_epoch<int> num_keep<int> num_2hop<int> num_rand<int> name<str>"
        exit 1
    fi
    seed=$2
    out_lim=$3
    num_msg=$4
    topo=$5
    end_epoch=$6
    num_keep=$7
    num_2hop=$8
    num_rand=$9
    name=${10}
    record_round=$(seq 0 $end_epoch)


    dirname="secHop-${name}-${out_lim}out-${num_msg}msg-${num_keep}keep-${num_2hop}secHop-${num_rand}rand-seed${seed}-${end_epoch}epoch"
    dirpath="output/$dirname"   
    rm -rf $dirpath
    mkdir -p $dirpath
    mkdir -p $dirpath/src
    cp *.py $dirpath/src

    echo $(date '+%Y-%m-%d %H:%M:%S')  > $dirpath/experiment.log
    echo "./run.sh $@" >> $dirpath/experiment.log


    cmd="python testbed.py run-2hop $seed $dirpath ${out_lim} ${num_msg} ${topo} ${num_keep} ${num_2hop} ${num_rand} ${record_round}"

    echo $cmd >> $dirpath/experiment.log
    run_command "$cmd"

    plot_cmd="python script/plot_single.py ${dirpath} $topo 90 node ${record_round}"

    echo $(date '+%Y-%m-%d %H:%M:%S')  >> $dirpath/experiment.log
    echo ${plot_cmd} >> $dirpath/experiment.log

    run_command "${plot_cmd}"
    echo "Run ->"
    echo "scp turing:/home/bowen/system-network/perigee-bandit-ml/${dirpath}/latest.png ."
}

function gen_rand_topo {
    if [ $# -ne 7 ]; then
        echo "./run.sh gen-rand-topo num_node<int> num_pub<int> name<str> proc_mean<float> proc_std<float> square_len<int>"
        exit 1
    fi
    num_node=$2
    num_pub=$3
    name=$4
    proc_mean=$5
    proc_std=$6
    square_len=$7
    ./script/gen-rand-topo.py ${num_node} ${num_pub} ${proc_mean} ${proc_std} ${square_len} > ./topo/${name}.json
    ./script/plot_topo.py ./topo/${name}.json ./topo/${name}
    echo "finish plot"

    echo "Run ->"
    echo "scp turing:/home/bowen/system-network/perigee-bandit-ml/topo/${name}.png . && open ${name}.png"
    exit 0
}

function gen_datacenter {
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
}

function compare {
    if [ $# -lt 3 ]; then
        echo "./run.sh compare x-percent<int0-100> percent-unit<node/hash> figname<str> exp1<str> exp2<str> .."
        exit 1
    fi
    figname=output/compare/$3
    python ./script/plot_compare.py $1 $2 $figname ${@:4}
    currpath=$(pwd)
    echo ${@:4}
    echo "Run ->"
    echo "scp turing:$currpath/$figname.png ."
}

function run_simple_model {
    python ./testbed.py run-simple-model 
}

subcommand=$1
case $subcommand in
    help)
        echo "./run.sh subcommand[run-mc/run-2hop/gen-rand-topo/gen-datacenter/compare/run-simple-model]" ;;
    run)
        run $@ ;;
    run-mc)
        run_mc $@ ;;
    run-2hop)
        run_2hop $@ ;;
    run-simple-model)
        run_simple_model ;;
    gen-rand-topo)
        gen_rand_topo $@ ;;
    gen-datacenter)
        gen_datacenter $@ ;;
    compare)
        compare ${@:2} ;;
    *)
        tput setaf 1
        echo "Unknown subcommand ${subcommand}"
        echo "./run.sh help"
        tput sgr0 ;;
esac
