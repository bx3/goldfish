#!/bin/bash
trap kill_procs INT

function kill_procs() {
    echo "STOP"
    for pid in $pids; do
        echo "kill process $pid"
        kill $pid
    done
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
        echo "./run.sh gen-rand-topo num_node<int> num_pub<int> name<str> proc_mean<float> proc_std<float> square_len<int> seed<int>"
        exit 1
    fi
    num_node=$1
    num_pub=$2
    name=$3
    proc_mean=$4
    proc_std=$5
    square_len=$6
    seed=$7

    ./script/gen-rand-topo.py ${num_node} ${num_pub} ${proc_mean} ${proc_std} ${square_len} ${seed} > ./topo/${name}.json
    ./script/plot_topo.py ./topo/${name}.json ./topo/${name} 0
    echo "finish plot"

    echo "Run ->"
    echo "scp turing:/home/bowen/system-network/perigee-bandit-ml/topo/${name}.png . && open ${name}.png"
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
    ./script/plot_topo.py ./topo/${name}.json ./topo/${name} 0
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

function run_batch_simple_model {
    if [ $# -ne 7 ]; then
        echo "./run.sh run-batch-simple-model start_seed<int> num_topo<int> num_node<int> num_pub<int> proc_mean<float> proc_std<float> square_len<int>"
        echo "Exmaple. ./run.sh run-batch-simple-model 15 5 100 3 20 0 500"
        exit 1
    fi
   
    start_seed=$1
    num_topo=$2
    num_node=$3
    num_pub=$4
    proc_mean=$5
    proc_std=$6
    square_len=$7
    end_seed=$(( ${start_seed} + ${num_topo} - 1 ))
    mkdir output-simple/batch-graph

    pids=""
    names=""
    for seed in $(seq ${start_seed} ${end_seed} ); do
        echo $seed
        name="rand-${num_node}node-${num_pub}pub-${proc_mean}_${proc_std}proc-${square_len}len-${seed}seed"
        gen_rand_topo ${num_node} ${num_pub} ${name} ${proc_mean} ${proc_std} ${square_len} ${seed} > /dev/null
        run_simple_model ${seed} topo/${name}.json n &
        pids="$pids $!"
        names="$names $name"
    done

    for pid in $pids; do
        wait $pid
    done

    # copy all results to dir
    for name in $names; do
        cp output-simple/$name/$name.png output-simple/batch-graph
    done
    echo "scp -r turing:/home/bowen/system-network/perigee-bandit-ml/output-simple/batch-graph ."
}

function run_simple_model {
    if [ $# -lt 3 ]; then
        echo "./run.sh run-simple-model seed<int> topo<str> print<y/n>"
        echo "Exmaple. ./run.sh run-simple-model 1 topo/rand-100node-3pub-20_0proc-500len-7seed.json n"
        exit 1
    fi
    seed=$1
    topo_json=$2
    printout=$3

    filename=$(basename -- "$topo_json")
    filename="${filename%.*}"
    
    rm -rf output-simple/${filename}
    mkdir -p output-simple/${filename}
    cp ${topo_json} output-simple/${filename}
    cp testbed.py output-simple/${filename}

    echo "python ./testbed.py run-simple-model $seed ${topo_json} output-simple/${filename}/${filename}" > "output-simple/${filename}/command.txt"
    echo "python ./script/plot_dists.py output-simple/${filename}/${filename}" >> "output-simple/${filename}/command.txt"
   
    echo "Started at: $(date)" 
    if [ $printout = 'y' ]; then
        python ./testbed.py run-simple-model $seed ${topo_json} output-simple/${filename}/${filename} 
    else
        python ./testbed.py run-simple-model $seed ${topo_json} output-simple/${filename}/${filename} > output-simple/${filename}/log.txt
    fi

    if [ "$?" -ne 0 ]; then
        echo "simulation bug. Exit"
        exit 1
    fi
    echo "Ended at: $(date)" 

    python ./script/plot_dists.py output-simple/${filename}/${filename}

    echo "Run ->"
    echo "vim output-simple/${filename}/log.txt"
    echo "scp turing:/home/bowen/system-network/perigee-bandit-ml/output-simple/${filename}/${filename}.png . && open ${filename}.png" | tee -a "output-simple/${filename}/command.txt"
}

subcommand=$1
case $subcommand in
    help)
        echo "./run.sh subcommand[run-mc/run-2hop/gen-rand-topo/gen-datacenter/compare/run-simple-model/run-batch-simple-model]" ;;
    run)
        run $@ ;;
    run-mc)
        run_mc $@ ;;
    run-2hop)
        run_2hop $@ ;;
    run-simple-model)
        run_simple_model ${@:2};;
    run-batch-simple-model)
        run_batch_simple_model ${@:2};;
    gen-rand-topo)
        gen_rand_topo ${@:2} ;;
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
