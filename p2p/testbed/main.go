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
	Rate float32 `json:"rate"`
	//Proc int `json:"proc"`
}

func run(args []string) {
	if len(args) != 5 {
		fmt.Println("./main run configFile<str> numOut<int> numIn<int> numMsg<int> numEpoch<int>")
		os.Exit(1)
	}
	configFile := args[0]
	numOut, _ := strconv.Atoi(args[1])
	numIn, _ := strconv.Atoi(args[2])
	numMsg, _ := strconv.Atoi(args[3])
	numEpoch, _ := strconv.Atoi(args[4])

	jsonFile, err := os.Open(configFile)
	if err != nil {
		log.Fatal(err)
	}
	data, err := ioutil.ReadAll(jsonFile)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(string(data))

	config := &ServerConfig{}
	err = json.Unmarshal([]byte(data), config)
	if err != nil {
		log.Fatal(err)
	}

	addrToId := make(map[string]int)
	for _, peer := range config.Peers {
		addrToId[peer.Addr] = peer.Id
	}

	pub := app.NewPublisher(config.Local.Rate, config.Local.Id, numEpoch, numMsg)
	goldfish := protocol.NewGoldfish(config.Local.Id, numOut, numIn, numMsg, numEpoch)
	server := network.StartServer(config.Local.Id, config.Local.Addr, numOut, numIn, addrToId, goldfish, pub)

	conns := make([]int, numOut)
	//gen := rand.New(rand.NewSource(time.Now().UnixNano()))
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
	fmt.Println(config.Local.Id, " connects to ", conns)
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
