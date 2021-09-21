#!/bin/bash
trap kill_procs INT

default_num_record=7
default_unit='hash'
default_cent=90
default_snapshot_dir='snapshots-exploit'

function kill_procs() {
    echo "STOP"
    for pid in $pids; do
        echo "kill process $pid"
        kill $pid
    done
}

#function run {
    #seed=$2
    #name=$3
    #out_lim=$4
    #num_region=$5
    #num_new=$6
    #use_node_hash=$7
    #input_json=$8
    #end_epoch=$9
    #record_round=$(seq 0 $end_epoch)

    #dirname="${name}-seed${seed}"
    #dirpath="analysis/$dirname"
    #mkdir $dirpath
    #cp *.py $dirpath
    #python testbed.py ${subcommand} ${seed} ${dirpath} ${out_lim} ${num_region} ${num_new} ${use_node_hash} ${input_json} ${record_round}
#}

function run_command {
    cmd=$1
    $cmd
    retval=$?
    if [ "$retval" -ne 0 ]; then
        echo "simulation bug. Exit"
        exit 1
    fi
}

#function run_mc {
    #if [ $# -ne 8 ]; then
        #echo "./run.sh run-mc seed<int> out_lim<int> num_new<int> topo<str> end_epoch<int> name<str> use_logger<y/n>"
        #exit 1
    #fi

    #seed=$2
    #out_lim=$3
    #num_new=$4
    #topo=$5
    #end_epoch=$6
    #name=$7
    #use_logger=$8
    #record_round=$(seq 0 $end_epoch)

    #dirname="${name}-${out_lim}out-${num_new}new-seed${seed}-${end_epoch}epoch"
    #dirpath="output/$dirname"
    #rm -rf $dirpath
    #mkdir -p $dirpath
    #mkdir -p $dirpath/src
    #cp *.py $dirpath/src

    #echo $(date '+%Y-%m-%d %H:%M:%S')  > $dirpath/experiment.log
    #echo "./run.sh $@" >> $dirpath/experiment.log

    #cmd="python testbed.py ${subcommand} ${seed} ${dirpath} ${out_lim} ${num_new} ${topo} ${use_logger} ${record_round}"
    #echo $cmd >> $dirpath/experiment.log

    #run_command "$cmd"
    
    #plot_cmd="python script/plot_single.py ${dirpath} $topo 90 node ${record_round}"

    #echo $(date '+%Y-%m-%d %H:%M:%S')  >> $dirpath/experiment.log
    #echo ${plot_cmd} >> $dirpath/experiment.log

    #run_command "${plot_cmd}"
    #echo "Run ->"
    #echo "scp turing:/home/bowen/system-network/goldfish/sim/${dirpath}/latest.png ."
#}


function run_2hop {
    if [ $# -ne 6 ]; then
        echo "./run.sh run-2hop seed<int> topo<str> num_epoch<int> num_adapt<int> churn_rate<float> context<str> "
        echo "Example. ./run.sh run-2hop 12 topo/rand-100node-3pub-20_0proc-500len-12seed.json 32 1 0.25 default "
        exit 1
    fi
    seed=$1
    topo_json=$2
    num_epoch=$3
    num_adapt=$4
    churn_rate=$5
    context=${6%/}

    end_epoch=$(( ${num_epoch} - 1 ))
    plot_interval=$(( ${end_epoch} / $default_num_record ))
    record_round=''
    for (( COUNTER=0; COUNTER<=${end_epoch}; COUNTER+=${plot_interval} )); do
        record_round="$record_round $COUNTER"
    done
    if [ $COUNTER -ne ${end_epoch} ]; then 
        record_round="${record_round} ${end_epoch}"
    fi

    filename=$(basename -- "$topo_json")
    filename="${filename%.*}"
    filename="$filename-${num_adapt}stars"

    rm -rf output-2hop/$context/${filename}
    mkdir -p output-2hop/$context/${filename}
    mkdir -p output-2hop/$context/${filename}/snapshots
    mkdir -p output-2hop/$context/${filename}/snapshots-exploit
    mkdir -p output-simple/${context}/${filename}/graphs


    cp ${topo_json} output-2hop/$context/${filename}
    cp ${topo_json} output-2hop/${context}/${filename}/topo.json
    cp testbed.py output-2hop/$context/${filename}


    echo "python ./testbed.py run-2hop $seed ${topo_json} ${num_epoch} ${num_adapt} ${churn_rate} output-2hop/$context/${filename}/${filename}" > "output-2hop/$context/${filename}/command.txt"
    echo "python ./script/plot_single.py output-2hop/$context/${filename} ${topo_json} ${default_cent} ${default_unit} ${default_snapshot_dir} " ${record_round} >> "output-2hop/$context/${filename}/command.txt"

    python ./testbed.py run-2hop $seed ${topo_json} ${num_epoch} ${num_adapt} ${churn_rate} output-2hop/$context/${filename}/${filename}

    if [ "$?" -ne 0 ]; then
        echo "simulation bug. Exit"
        exit 1
    fi

    python ./script/plot_dists.py output-2hop/$context/${filename}/${filename} ${topo_json} ${record_round}
    python ./script/plot_single.py output-2hop/$context/${filename} ${topo_json} ${default_cent} ${default_unit} ${default_snapshot_dir} ${record_round}
    python ./script/plot_node_cdf.py output-2hop/${context}/${filename} ${record_round}

    echo "Run ->"
    echo "vim output-2hop/$context/${filename}/log.txt"
    echo "scp turing:/home/bowen/system-network/goldfish/sim/output-2hop/$context/${filename}/${filename}.png . && open ${filename}.png" | tee -a "output-2hop/$context/${filename}/command.txt"
    echo "scp turing:/home/bowen/system-network/goldfish/sim/output-2hop/$context/${filename}/${filename}-lat${default_cent}-${default_unit}.png . && open ${filename}-lat${default_cent}-${default_unit}.png" | tee -a "output-2hop/$context/${filename}/command.txt"
}

function gen_real_topo {
    if [ $# -ne 7 ]; then 
        echo "./run.sh gen-real-topo num_node<int> dist<unif/exp/real>  num_pub<int> name<str> proc_mean<float> proc_std<float> seed<int>"
        exit 0
    fi
    num_node=$1
    num_pub=$2
    dist=$3
    name=$4
    proc_mean=$5
    proc_std=$6
    seed=$7
    server_meta='inputs/servers-2020-07-19.csv'
    ping_data='inputs/pings-2020-07-19-2020-07-20.csv'
    ./script/gen-real-topo.py ${server_meta} ${ping_data} $@ > ./topo/${name}.json
}

function gen_rand_topo {
    if [ $# -ne 8 ]; then
        echo "./run.sh gen-rand-topo num_node<int> num_pub<int> distribution<unif/exp> name<str> proc_mean<float> proc_std<float> square_len<int> seed<int>"
        exit 1
    fi
    num_node=$1
    num_pub=$2
    distribution=$3
    name=$4
    proc_mean=$5
    proc_std=$6
    square_len=$7
    seed=$8

    ./script/gen-rand-topo.py ${num_node} ${num_pub} ${distribution} ${proc_mean} ${proc_std} ${square_len} ${seed} > ./topo/${name}.json
    ./script/plot_topo.py ./topo/${name}.json ./topo/${name} 0
    echo "finish plot"

    echo "Run ->"
    echo "scp turing:/home/bowen/system-network/goldfish/sim/topo/${name}.png . && open ${name}.png"
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
    echo "scp turing:/home/bowen/system-network/goldfish/sim/topo/${name}.png . && open ${name}.png"
}

function compare_batch {
    if [ $# -lt 12 ]; then
        echo "./run.sh compare-batch x-percent<int0-100> percent-unit<node/pub/hash> batch1<str> batch2<str> start_seed<int> num_seed<int> num_node<int> num_pub<int> num_star<int> snapshots_dir<snapshots/snapshots-exploit> topo_type<real/rand> epochs<int list>"
        echo "Example. ./run.sh compare-batch 90 pub output-simple/ output-2hop/ 15 15 100 3 1 snapshots-exploit 0 2 4 8 16 31" 
        exit 1
    fi
    x_cent=$1
    unit=$2
    batch1="${3%/}"
    batch2="${4%/}"
    start_seed=$5
    num_seed=$6
    num_node=$7
    num_pub=$8
    num_star=$9
    snapshots_dir=${10}
    topo_type=${11}
    record_epochs=${@:12}
    proc_mean=20
    proc_std=0
    square_len=500

    end_seed=$(( ${start_seed} + ${num_seed} - 1 ))
    compare_name="${num_pub}pubs-${num_star}stars"
    rm -rf output-compare/${compare_name}
    mkdir output-compare/${compare_name}

    pids=''
    for seed in $(seq ${start_seed} ${end_seed} ); do
        name="${topo_type}-${num_node}node-${num_pub}pub-${proc_mean}_${proc_std}proc-${seed}seed"
        simple_model_path="${batch1}/$name-${num_star}stars"
        sec_hop_path="${batch2}/$name-${num_star}stars"
        if [[ ! -d "${simple_model_path}" ]]; then
            echo "Error. Missing ${simple_model_path}"
            exit 1
        fi
        if [[ ! -d "${sec_hop_path}" ]]; then
            echo "Error. Missing ${sec_hop_path}"
            exit 1
        fi

        check_init_setup ${simple_model_path} ${sec_hop_path} $name
        topo_json="${simple_path}/${setup_name}.json"

        figname="output-compare/${compare_name}/${name}"
        compare ${x_cent} ${unit} $figname ${simple_model_path} ${sec_hop_path} ${topo_json} ${snapshots_dir} ${record_epochs} &
        pids="$pids $!"
    done

    for pid in $pids; do
        wait $pid
    done
    
    echo "scp -r turing:/home/bowen/system-network/goldfish/sim/output-compare/${compare_name} ."
}

function check_init_setup {
    simple_path=$1
    sec_hop_path=$2
    setup_name=$3
    
    if [ ! -f "${simple_path}/adapts" ]; then
        echo "Missing. ${simple_path}/adapts"
        exit 1
    fi
    if [ ! -f "${sec_hop_path}/adapts" ]; then
        echo "Missing. ${sec_hop_path}/adapts"
        exit 1
    fi
    if [ ! -f "${simple_path}/topo.json" ]; then
        echo "Missing. ${simple_path}/topo.json"
        exit 1
    fi
    if [ ! -f "${sec_hop_path}/topo.json" ]; then
        echo "Missing. ${sec_hop_path}/topo.json"
        exit 1
    fi
    if [ ! -f "${simple_path}/init.json" ]; then
        echo "Missing. ${simple_path}/init.json"
        exit 1
    fi
    if [ ! -f "${sec_hop_path}/init.json" ]; then
        echo "Missing. ${sec_hop_path}/init.json"
        exit 1
    fi

    cmp -s "${simple_path}/adapts" "${sec_hop_path}/adapts"
    result=$?
    # identical adapts node
    if [ ${result} -ne 0 ]; then
        echo "Error. Adapt nodes are diff. "
        echo ${simple_path}/adapts
        echo ${sec_hop_path}/adapts
        exit 1
    fi

    cmp -s "${simple_path}/topo.json" "${sec_hop_path}/topo.json"
    topo_result=$?
    if [ ${topo_result} -ne 0 ]; then
        echo "Error. topology json file are diff."
        echo ${simple_path}/${setup_name}.json
        echo ${sec_hop_path}/${setup_name}.json
        exit 1
    fi

    cmp -s "${simple_path}/init.json" "${sec_hop_path}/init.json"
    init_result=$?
    # identical init conn json
    if [ ${init_result} -ne 0 ]; then
        echo "Error. init network connections are diff. "
        echo ${simple_path}
        echo ${sec_hop_path}
        exit 1
    fi
}

function compare {
    if [ $# -lt 8 ]; then
        echo "./run.sh compare x-percent<int(0-100)/avg> percent-unit<node/pub/hash> figname<str> exp1<str> exp2<str> topo<str> snapshots_dir<snapshots/snapshots-exploit> epochs<int list>"
        exit 1
    fi
    cent=$1
    unit=$2
    figname=$3
    exp1=$4
    exp2=$5
    topo_json=$6
    snapshot_dir=$7
    record_list=${@:8}

    python ./script/plot_compare.py $@
    if [ "$?" -ne 0 ]; then
        echo "simulation bug. Exit"
        exit 1
    fi 


    #cvg_name="compare-batch/${compare_name}/${name}-box-cvg"
    #python script/plot_convergence.py ${cvg_name} $cent $unit ${simple_model_path} ${sec_hop_path} 
    cdfs_name="output-compare/${compare_name}/${name}-cdfs"
    python script/plot_compare_cdfs.py ${cdfs_name} ${exp1} ${exp2} ${record_list}

    currpath=$(pwd)
    echo "Run ->"
    echo "scp turing:$currpath/$3.png ."
}

function run_batch_2hop {
    if [ $# -ne 12 ]; then
        echo "./run.sh run-batch-2hop start_seed<int> num_topo<int> num_node<int> num_pub<int> dist<unif/exp> num_adapt<int> proc_mean<float> proc_std<float> topo_type<real/rand> num_epoch<int> churn_rate<float> context<str>"
        echo "Exmaple. ./run.sh run-batch-2hop 15 15 100 3 1 unif 20 0 rand 500 0.25 default"
        exit 1
    fi
    start_seed=$1
    num_topo=$2
    num_node=$3
    num_pub=$4
    num_adapt=$5
    dist=$6
    proc_mean=$7
    proc_std=$8
    topo_type=$9
    num_epoch=${10}
    churn_rate=${11}
    context=${12}
    end_seed=$(( ${start_seed} + ${num_topo} - 1 ))
    rm -rf output-2hop/${context}-context-summary
    mkdir output-2hop/${context}-context-summary
    rm -rf output-2hop/${context}-context
    mkdir output-2hop/${context}-context

    pids=""
    names=""
    for seed in $(seq ${start_seed} ${end_seed} ); do
        echo $seed
        #name="${topo_type}-${num_node}node-${num_pub}pub-${proc_mean}_${proc_std}proc-${seed}seed"
        name="${topo_type}-${num_node}node-${num_pub}pub-${dist}-${proc_mean}_${proc_std}proc-${seed}seed"

        if [[ -f "topo/$name.json" ]]; then 
            run_2hop $seed topo/${name}.json ${num_epoch} ${num_adapt} ${churn_rate} ${context}-context &
            pids="$pids $!"
            names="$names $name"
        else
           echo "$name.json does not exist"
           exit 1
        fi 
    done

    for pid in $pids; do
        wait $pid
    done

    for name in $names; do
        exp_name="$name-${num_adapt}stars"
        cp output-2hop/${context}-context/${exp_name}/${exp_name}.png output-2hop/${context}-context-summary
        cp output-2hop/${context}-context/${exp_name}/${exp_name}-lat${default_cent}-${default_unit}.png output-2hop/${context}-context-summary

        cp output-2hop/${context}-context/${exp_name}/cdfs.png output-2hop/${context}-context-summary/${exp_name}-cdfs.png


    done
    echo "scp -r turing:/home/bowen/system-network/goldfish/sim/output-2hop/${context}-context-summary . && open ${context}-context-summary/*"

}

function run_batch_simple_model {
    if [ $# -ne 13 ]; then
        echo "./run.sh run-batch-simple-model start_seed<int> num_topo<int> num_node<int> num_pub<int> num_star<int> dist<unif/exp/real> proc_mean<float> proc_std<float> topo_type<real/rand> num_epoch<int> churn_rate<float> para<y/n> context<str>"
        echo "Exmaple. ./run.sh run-batch-simple-model 15 5 100 3 1 exp 20 0 500 32 0.25 y default"
        exit 1
    fi
   
    start_seed=$1
    num_topo=$2
    num_node=$3
    num_pub=$4
    num_star=$5
    dist=$6
    proc_mean=$7
    proc_std=$8
    topo_type=$9
    num_epoch=${10}
    churn_rate=${11}
    use_para=${12}
    context=${13}
    end_seed=$(( ${start_seed} + ${num_topo} - 1 ))
    #rm -rf output-simple/${context}-context-summary
    #mkdir output-simple/${context}-context
    square_len=500
    

    rm -rf output-simple/${context}-context-summary
    mkdir output-simple/${context}-context-summary

    pids=""
    names=""
    if [ ${use_para} = 'y' ]; then 
        for seed in $(seq ${start_seed} ${end_seed} ); do
            echo $seed
            name="${topo_type}-${num_node}node-${num_pub}pub-${dist}-${proc_mean}_${proc_std}proc-${seed}seed"
            if [ ${topo_type} = 'real' ]; then
                gen_real_topo ${num_node} ${num_pub} ${dist} ${name} ${proc_mean} ${proc_std} ${seed} > /dev/null
            elif [ ${topo_type} = 'rand' ]; then
                gen_rand_topo ${num_node} ${num_pub} ${dist} ${name} ${proc_mean} ${proc_std} ${square_len} ${seed} > /dev/null
            else
                echo "Unknown topo-type ${topo_type}"
                exit 1
            fi
            
            run_simple_model ${seed} topo/${name}.json ${num_star} ${num_epoch} ${churn_rate} n ${context}-context &
            pids="$pids $!"
            names="$names $name"
        done

        for pid in $pids; do
            wait $pid
        done
    else
        for seed in $(seq ${start_seed} ${end_seed} ); do
            echo $seed
            name="${topo_type}-${num_node}node-${num_pub}pub-${proc_mean}_${proc_std}proc-${seed}seed"
            if [ ${topo_type} = 'real' ]; then
                gen_real_topo ${num_node} ${num_pub} ${dist} ${name} ${proc_mean} ${proc_std} ${seed} > /dev/null
            elif [ ${topo_type} = 'rand' ]; then
                gen_rand_topo ${num_node} ${num_pub} ${dist} ${name} ${proc_mean} ${proc_std} ${square_len} ${seed} > /dev/null
            else
                echo "Unknown topo-type ${topo_type}"
                exit 1
            fi

            run_simple_model ${seed} topo/${name}.json ${num_star} ${num_epoch} ${churn_rate} ${context}-context n
            names="$names $name"
        done
    fi

    # copy all results to dir
    for name in $names; do
        exp_name="$name-${num_star}stars"
        cp output-simple/${context}-context/${exp_name}/${exp_name}.png output-simple/${context}-context-summary
        cp output-simple/${context}-context/${exp_name}/${exp_name}-lat${default_cent}-${default_unit}.png output-simple/${context}-context-summary
        cp output-simple/${context}-context/${exp_name}/cdfs.png output-simple/${context}-context-summary/${exp_name}-cdfs.png
    done
    echo "scp -r turing:/home/bowen/system-network/goldfish/sim/output-simple/${context}-context-summary . && open ${context}-context-summary/*"
}

function replot_dir {
    if [ $# -ne 7 ]; then
        echo "./run.sh replot-dir context<str> start_seed<int> num_topo<int> num_epoch<int> x_cent<int(0-100)/avg> unit<node/pub/hash> snapshots_dir<snapshots/snapshots-exploit>"
        echo "Exmaple. ./run.sh replot-dir output-simple/default 15 5 128 90 pub snapshots-exploit"
        exit 1
    fi
    context="${1%/}"
    start_seed=$2
    num_topo=$3
    num_epoch=$4
    x_cent=$5
    unit=$6
    snapshot_dir=$7

    end_seed=$(( ${start_seed} + ${num_topo} - 1 ))
    end_epoch=$(( ${num_epoch} - 1 ))
    plot_interval=$(( ${end_epoch} / $default_num_record ))
    record_round=''
    for (( COUNTER=0; COUNTER<=${end_epoch}; COUNTER+=${plot_interval} )); do
        record_round="$record_round $COUNTER"
    done
    if [ $COUNTER -ne ${end_epoch} ]; then 
        record_round="${record_round} ${end_epoch}"
    fi

    names=''
    pids=''
    files=$(ls ${context})
    for name in ${files}; do
        names="$names $name"
        topo_json="${context}/${name}/topo.json"
        if [ -f ${context}/${name}/${name} ]; then
            python ./script/plot_dists.py ${context}/${name}/${name} ${topo_json} ${record_round} &
            pids="$pids $!"
        fi
        python ./script/plot_single.py ${context}/${name} ${topo_json} ${x_cent} $unit ${snapshot_dir} ${record_round} &
        pids="$pids $!"
        python ./script/plot_node_cdf.py ${context}/${name} ${record_round} &
        pids="$pids $!"
    done


    for pid in $pids; do
        wait $pid
    done
    
    rm -rf ${context}-summary-replot
    mkdir ${context}-summary-replot
    for name in $names; do
        cp ${context}/${name}/${name}.png ${context}-summary-replot
        #cp ${context}/${name}/${name}-lat${x_cent}-${unit}.png ${context}-summary-replot
        cp ${context}/${name}/cdfs.png ${context}-summary-replot/${name}-cdfs.png
    done
    echo "scp -r turing:/home/bowen/system-network/goldfish/sim/${context}-summary-replot ."
}

function replot_batch {
    if [ $# -ne 12 ]; then
        echo "./run.sh replot-batch context<str> start_seed<int> num_topo<int> num_node<int> num_pub<int> num_star<int> proc_mean<float> proc_std<float> square_len<int> num_epoch<int> x_cent<int(0-100)/avg> unit<node/pub/hash>"
        echo "Exmaple. ./run.sh replot-batch output-simple 15 5 100 3 1 20 0 500 128 90 pub"
        exit 1
    fi
    context="${1%/}"
    start_seed=$2
    num_topo=$3
    num_node=$4
    num_pub=$5
    num_star=$6
    proc_mean=$7
    proc_std=$8
    square_len=$9
    num_epoch=${10}
    x_cent=${11}
    unit=${12}

    end_seed=$(( ${start_seed} + ${num_topo} - 1 ))
    end_epoch=$(( ${num_epoch} - 1 ))
    plot_interval=$(( ${end_epoch} / $default_num_record ))
    record_round=''
    for (( COUNTER=0; COUNTER<=${end_epoch}; COUNTER+=${plot_interval} )); do
        record_round="$record_round $COUNTER"
    done
    if [ $COUNTER -ne ${end_epoch} ]; then 
        record_round="${record_round} ${end_epoch}"
    fi

    replot_dir=${context}-summary-replot
    rm -rf ${replot_dir}
    mkdir ${replot_dir}

    names=''
    pids=''
    for seed in $(seq ${start_seed} ${end_seed} ); do
        filename="rand-${num_node}node-${num_pub}pub-${proc_mean}_${proc_std}proc-${square_len}len-${seed}seed-${num_star}stars"
        names="$names $filename"
        topo_name="rand-${num_node}node-${num_pub}pub-${proc_mean}_${proc_std}proc-${square_len}len-${seed}seed"
        topo_json="${context}/$filename/${topo_name}.json"
        if [ -f ${context}/${filename}/${filename} ]; then 
            python ./script/plot_dists.py ${context}/${filename}/${filename} ${topo_json} ${record_round} &
            pids="$pids $!"
        fi
        python ./script/plot_single.py ${context}/${filename} ${topo_json} ${x_cent} $unit ${default_snapshot_dir} ${record_round} &
        pids="$pids $!"
    done 

    for pid in $pids; do
        wait $pid
    done
    
    rm -rf ${context}-summary-replot
    mkdir ${context}-summary-replot
    for name in $names; do
        cp ${context}/${name}/${name}.png ${context}-summary-replot
        cp ${context}/${name}/${name}-lat${x_cent}-${unit}.png ${context}-summary-replot
    done
    echo "scp -r turing:/home/bowen/system-network/goldfish/sim/${context}-summary-replot ."
}

function run_simple_model {
    if [ $# -ne 7 ]; then
        echo "./run.sh run-simple-model seed<int> topo<str> num_star<int> num_epoch<int> churn_rate<float> print<y/n> context<str>"
        echo "Exmaple. ./run.sh run-simple-model 1 topo/rand-100node-3pub-20_0proc-500len-7seed.json 10 30 0.25 n default"
        exit 1
    fi
    seed=$1
    topo_json=$2
    num_star=$3
    num_epoch=$4
    churn_rate=$5
    printout=$6
    context=$7

    end_epoch=$(( ${num_epoch} - 1 ))
    plot_interval=$(( ${end_epoch} / $default_num_record ))
    if [ ${plot_interval} -eq 0 ]; then
        echo "${plot_interval} is too small. Increase epoch "
        exit 1
    fi
    
    record_round=''
    for (( COUNTER=0; COUNTER<=${end_epoch}; COUNTER+=${plot_interval} )); do
        record_round="$record_round $COUNTER"
    done
    if [ $COUNTER -ne ${end_epoch} ]; then 
        record_round="${record_round} ${end_epoch}"
    fi

    filename=$(basename -- "$topo_json")
    filename="${filename%.*}"
    filename="$filename-${num_star}stars"

    rm -rf output-simple/${context}/${filename}
    mkdir -p output-simple/${context}/${filename}
    mkdir -p output-simple/${context}/${filename}/snapshots
    mkdir -p output-simple/${context}/${filename}/snapshots-exploit
    mkdir -p output-simple/${context}/${filename}/graphs


    cp ${topo_json} output-simple/${context}/${filename}
    cp ${topo_json} output-simple/${context}/${filename}/topo.json
    cp testbed.py output-simple/${context}/${filename}

    startDate=$(date +%s) 
    echo "Started at: $(date)" > "output-simple/${context}/${filename}/command.txt"
    echo "./run.sh run-simple-model $@" >> "output-simple/${context}/${filename}/command.txt"
    echo "python ./testbed.py run-simple-model $seed ${topo_json} output-simple/${context}/${filename}/${filename} ${num_epoch} ${churn_rate} $printout" >> "output-simple/${context}/${filename}/command.txt"
    echo "python ./script/plot_dists.py output-simple/${context}/${filename}/${filename} ${topo_json}" ${record_round} >> "output-simple/${context}/${filename}/command.txt"
    echo "python ./script/plot_single.py output-simple/${context}/${filename} ${topo_json} ${default_cent} ${default_unit} ${default_snapshot_dir}" ${record_round} >> "output-simple/${context}/${filename}/command.txt"
    echo "python ./script/plot_node_cdf.py output-simple/${context}/${filename} " ${record_round} >> "output-simple/${context}/${filename}/command.txt"


    if [ $printout = 'y' ]; then
        python ./testbed.py run-simple-model $seed ${topo_json} output-simple/${context}/${filename}/${filename} ${num_star} ${num_epoch} ${churn_rate} y
    else
        python ./testbed.py run-simple-model $seed ${topo_json} output-simple/${context}/${filename}/${filename} ${num_star} ${num_epoch} ${churn_rate} n 
        #> output-simple/${filename}/log.txt
    fi

    if [ "$?" -ne 0 ]; then
        echo "simulation bug. Exit"
        exit 1
    fi
    echo "Ended at: $(date)" >> "output-simple/${context}/${filename}/command.txt"
    endDate=$(date +%s)
    runtime=$(( $endDate - $startDate ))
    echo "Runtime: $runtime sec" >> "output-simple/${context}/${filename}/command.txt"

    python ./script/plot_dists.py output-simple/${context}/${filename}/${filename} ${topo_json} ${record_round}
    python ./script/plot_single.py output-simple/${context}/${filename} ${topo_json} ${default_cent} ${default_unit} ${default_snapshot_dir} ${record_round}
    python ./script/plot_node_cdf.py output-simple/${context}/${filename} ${record_round}

    echo "Run ->"
    echo "vim output-simple/${context}/${filename}/log.txt"
    echo "scp turing:/home/bowen/system-network/goldfish/sim/output-simple/${context}/${filename}/${filename}.png . && open ${filename}.png" | tee -a "output-simple/${context}/${filename}/command.txt"
    echo "scp turing:/home/bowen/system-network/goldfish/sim/output-simple/${context}/${filename}/${filename}-lat${default_cent}-${default_unit}.png . && open ${filename}-lat${default_cent}-${default_unit}.png" | tee -a "output-simple/${context}/${filename}/command.txt"
}

subcommand=$1
case $subcommand in
    help)
        echo "./run.sh subcommand[run-mc/run-2hop/gen-rand-topo/gen-real-topo/gen-datacenter/compare/run-simple-model/run-batch-simple-model/run-batch-2hop/compare-batch/replot-batch]" ;;
    run)
        run $@ ;;
    run-mc)
        run_mc $@ ;;
    run-2hop)
        run_2hop ${@:2} ;;
    run-batch-2hop)
        run_batch_2hop ${@:2} ;;
    run-simple-model)
        run_simple_model ${@:2};;
    run-batch-simple-model)
        run_batch_simple_model ${@:2};;
    gen-rand-topo)
        gen_rand_topo ${@:2} ;;
    gen-real-topo)
        gen_real_topo ${@:2} ;;
    gen-datacenter)
        gen_datacenter $@ ;;
    compare-batch)
        compare_batch ${@:2} ;;
    compare)
        compare ${@:2} ;;
    replot-batch)
        replot_batch ${@:2} ;;
    replot-dir)
        replot_dir ${@:2} ;;
    *)
        tput setaf 1
        echo "Unknown subcommand ${subcommand}"
        echo "./run.sh help"
        tput sgr0 ;;
esac
