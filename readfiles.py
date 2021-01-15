#!/usr/bin/env python
import numpy as np
import config
        
# The LinkDelay is the edge latency, can be generated by 'GenerateWeightFile.py'
def ReadWeightFile(num_node):
    WeightFileName=config.link_file
    LinkDelay=np.zeros([num_node,num_node])
    filew=open(WeightFileName,'r',errors='replace')
    line=filew.readlines()
    for i in range(num_node):
        a=line[i].strip().split("  ")
        # print(a)
        for j in range(num_node):
            LinkDelay[i][j]=a[j]
    filew.close()
    return(LinkDelay)

# The NodeHash are the edge latencies, can be generated by 'GenerateHashFile.py', the mean is 1
def ReadHashFile(num_node):
    HashFileName = config.hash_file 
    NodeHash=np.zeros(num_node)
    fileh=open(HashFileName,'r',errors='replace')
    line=fileh.readlines()
    a=line[0].split("  ")
    for j in range(num_node):
        NodeHash[j]=a[j]
    fileh.close()
    return(NodeHash)

# Select 20 nodes to build a low-latency subgraph with 10%'s latency but occupys 90%'s hash
# def ReadLowLatencyNode(a,NodeHash,delay,LinkDelay):
    # LowLatenncyNodeFileName="node"+str(a)+".txt"
    # fileh=open(LowLatenncyNodeFileName,'r',errors='replace')
    # line=fileh.readlines()
    # a=line[0].split("  ")
    # lowlatencynode=np.zeros(100)
    # for i in range(len(a)-1):
        # lowlatencynode[i]=int(a[i])-1
        # if int(lowlatencynode[i])<test_num:
            # delay[int(lowlatencynode[i])]      =   0.1*delay[int(lowlatencynode[i])]
            # NodeHash[int(lowlatencynode[i])] =   81*NodeHash[int(lowlatencynode[i])]
    # for i in range(len(lowlatencynode)):
        # for j in range(len(lowlatencynode)):
            # LinkDelay[i][j]=0.1*LinkDelay[i][j]
    # keep mean hash as 1
    # NodeHash=NodeHash/9
    # fileh.close()
    # return(NodeHash,delay,LinkDelay)
    


def Read(NodeDelay, NetworkType, num_node):
    LinkDelay= ReadWeightFile(num_node)
    # print(LinkDelay)
    #LinkDelay=   initnetwork.DelayByBandwidth(NeighborSets,bandwidth)
    NodeHash  = ReadHashFile(num_node)
    # if  str(NetworkType) == 'lowlatencyhash':
        # [NodeHash,NodeDelay, LinkDelay]  =   ReadLowLatencyNode(str(sys.argv[1]),NodeHash,NodeDelay,LinkDelay)
    # if  str(NetworkType) == 'treehash':
        # [LinkDelay,NodeDelay] =   ReadTreeEdge(str(sys.argv[1]),LinkDelay,NodeDelay)

    return(LinkDelay,NodeHash,NodeDelay)
