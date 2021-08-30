package app

import (
	"time"
	"math/rand"
	"github.com/bx3/goldfish/p2p/protocol"
)

type Publisher struct {
	gen *rand.Rand
	Rate float32
	Count int
	PubID int
	NumEpoch int
	NumMsg int
}

func NewPublisher(rate float32, pubID int, numEpoch int, numMsg int) (*Publisher) {
	return &Publisher {
		gen: rand.New(rand.NewSource(time.Now().UnixNano())),
		Rate: rate,
		Count: 0,
		PubID: pubID,
		NumEpoch: numEpoch,
		NumMsg: numMsg,
	}
}

func (p *Publisher) Publish() (bool, protocol.Message){
	n := p.gen.Float32()

	if p.Count > p.NumMsg * p.NumEpoch {
		return false, protocol.Message{}
	}

	if n < p.Rate {
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
