# goldfish Simulator

Run `./run.sh help` to list all subcommands.

Run multiple Goldfish simulations
`./run.sh run-batch-simple-model help`

Run multiple Perigee simulations
`./run.sh run-batch-2hop help`

For example, the following commands

`./run.sh run-batch-simple-model 15 10 100 3 10 unif 20 0 rand 100 1 y default`

creates 10 simulation starting at seed 15; each simulation contains 100 nodes (3publisher and 10adaptation node), the publishing probability is uniformly distributed; the simulation runs for 100 epochs, such that every adaptation node is trying to adapt at every epoch. 'y' allows multi-threads for 10 adaptation nodes; each of 10 simulations creates an output directory, and 10 of such directory is stored inside "default" directory created inside "output-simple/".

# Run following commands to reproduce the Evalutaions in the paper:

## For histogram of optimal connections, (fig2) 
`./run.sh run-batch-simple-model 15 200 100 3 1 unif 20 0 rand 300 1 y rand-100node-3pub-1star-20_0proc-500len-2update-3topo
then run`\
`./script/eval_optimal.py 15 200 300 0.05 output-simple/rand-100node-3pub-1star-20_0proc-500len-2update-3topo-context rand-100node-3pub-20_0proc`

## For bar plot
`./run.sh run-batch-simple-model 15 10 100 100 32 exp 20 0 rand 100 1 y rand-100node-expPub-32star-20_0proc-500len-2update-3topo`\
then
`./run.sh run-batch-2hop 15 10 100 100 32 exp 20 0 rand 100 1.0 rand-100node-expPubs-32stars-rand-out4-exploit3-churn100`\
then
`./script/plot_error_bar.py 15 10 32 output-simple/rand-100node-expPub-32star-20_0proc-500len-2update-3topo-context output-2hop/rand-100node-expPubs-32stars-rand-out4-exploit3-churn100-context rand-100node-100pub-exp-20_0proc 90 0 4 8 16 32 99`\

## For Generating table 
Repeat `./run.sh run-batch-simple-model` and `./run.sh run-batch-2hop` for all scenario, then extract the number by running the following
`./script/extract_dist.py 15 10 32  output-simple/rand-100node-32pub-32star-20_0proc-500len-2update-3topo-context rand-100node-32pub-20_0proc 50 99`
