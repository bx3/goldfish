package main

import (
	"fmt"
	"os"
	"log"
	"strconv"
	//"time"
	"io/ioutil"
	"math/rand"
	"encoding/json"
	"github.com/bx3/goldfish/p2p/network"
	"github.com/bx3/goldfish/p2p/protocol"
	"github.com/bx3/goldfish/p2p/app"
)

type ServerConfig struct {
	Local ServerID `json:"local"`
	Peers []ServerID `json:"peers"`
}

type ServerID struct {
	Id int `json:"id"`
	Addr string `json:"addr"`
	Prob float32 `json:"prob"`
    Adapt bool `json:"adapt"`
}

func run(args []string) {
	if len(args) != 7 {
		fmt.Println("./main run configFile<str> numOut<int> numIn<int> numRand<int> numMsg<int> numEpoch<int> rate-per-sec<int>")
        fmt.Println(args)
		os.Exit(1)
	}
	configFile := args[0]
	numOut, _ := strconv.Atoi(args[1])
	numIn, _ := strconv.Atoi(args[2])
    numRand, _ := strconv.Atoi(args[3])
	numMsg, _ := strconv.Atoi(args[4])
	numEpoch, _ := strconv.Atoi(args[5])
    rate, _ := strconv.Atoi(args[6])

    updateInterval := 1
    numTopo := 2

	jsonFile, err := os.Open(configFile)
	if err != nil {
		log.Fatal(err)
	}
	data, err := ioutil.ReadAll(jsonFile)
	if err != nil {
		log.Fatal(err)
	}

	config := &ServerConfig{}
	err = json.Unmarshal([]byte(data), config)
	if err != nil {
		log.Fatal(err)
	}

	idToAddr := make(map[int]string)
	for _, peer := range config.Peers {
		idToAddr[peer.Id] = peer.Addr
	}

    control := make(chan protocol.Signal)

	goldfish := protocol.NewGoldfish(configFile, config.Local.Adapt, config.Local.Id, numOut, numIn, numRand, numMsg, numEpoch, updateInterval, numTopo, control)
	server := network.StartServer(config.Local.Id, config.Local.Addr, numOut, numIn, idToAddr, goldfish, control)
    pub := app.NewPublisher(config.Local.Id, config.Local.Prob, rate, config.Local.Id, numEpoch, numMsg, server)

	conns := make([]int, numOut)
	gen := rand.New(rand.NewSource(int64(config.Local.Id)))
	for i, p := range gen.Perm(len(config.Peers)) {
		if i < numOut {
			peer := config.Peers[p]
			server.Connect(peer.Id, peer.Addr)
			conns[i] = peer.Id
		} else {
			break
		}
	}
    fmt.Println(config.Local.Id, "->", conns)
    pub.Run()
	select{}
}

func main() {
	switch os.Args[1] {
	case "run":
		run(os.Args[2:])
	default:
		fmt.Println("unknown subcommand", os.Args[1])
	}
}
