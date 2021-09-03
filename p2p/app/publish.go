package app

import (
    "fmt"
	"time"
	"math/rand"
	"github.com/bx3/goldfish/p2p/protocol"
    "github.com/bx3/goldfish/p2p/network"
)

type Publisher struct {
    id int
	gen *rand.Rand
    Prob float32
	Rate int  // every 1 sec, number msg
	Count int
	PubID int
	NumEpoch int
	NumMsg int
    server *network.Server
}

func NewPublisher(id int, prob float32, rate int, pubID int, numEpoch int, numMsg int, server *network.Server) (*Publisher) {
	return &Publisher {
        id: id,
		gen: rand.New(rand.NewSource(time.Now().UnixNano())),
        Prob: prob,
		Rate: rate,
		Count: 0,
		PubID: pubID,
		NumEpoch: numEpoch,
		NumMsg: numMsg,
        server: server,
	}
}

func (p *Publisher) Publish() (bool, protocol.Message){
	n := p.gen.Float32()
	if p.Count > p.NumMsg * p.NumEpoch {
		return false, protocol.Message{}
	}

	if n < p.Prob {
		fishMsg := protocol.FishMessage {
			PubID: p.PubID,
			PubCount: p.Count,
			PubTime: time.Now().UnixNano(),
		}
		msg := protocol.Message {
			Payload: fishMsg,
		}
		p.Count += 1
		return true, msg
	} else {
		return false, protocol.Message{}
	}
}

func (p *Publisher) Run() {
    msgFreq := time.Duration(int64(1000 / p.Rate))
    msgTicker := time.NewTicker(msgFreq * time.Millisecond)
    //topoFreq := time.Duration(int64(1000 / p.Rate * p.NumMsg))
    //topoTicker := time.NewTicker(topoFreq * time.Millisecond)
	done := make(chan bool)
	go func() {
		time.Sleep(1000 * time.Millisecond)
        // while it is adapting, messages are not sent for demo simplicity
		for {
			select {
			case <-msgTicker.C:
				ok, msg := p.Publish()
				if ok {
					fmt.Println(p.id, "Publish")
					p.server.Broadcast(msg)
				}
            //case <- topoTicker.C:
                //fmt.Println(p.id, "Adapt", time.Now())
                //p.server.AdaptTopo()
			case <-done:
				fmt.Println("done generating source msg")
				return
			}
		}
	}()
}
