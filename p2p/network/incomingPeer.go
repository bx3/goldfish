package network

import (
	//"fmt"
	"net"
	"encoding/gob"
	"log"
	"github.com/bx3/goldfish/p2p/protocol"
)

type IncomingPeer struct{
	localID int
	peerID int
	conn net.Conn
	enc *gob.Encoder
	dec *gob.Decoder
	incoming chan protocol.Message
	sendQueue chan protocol.Message
}

func NewIncomingPeer(localID int, conn net.Conn, incoming chan protocol.Message) *IncomingPeer{
	enc := gob.NewEncoder(conn)
	dec := gob.NewDecoder(conn)

	var msg protocol.Message
	err := dec.Decode(&msg)
	if err != nil {
		log.Fatal(err)
	}
	var peerID int
	if msg.IsInit {
		peerID = msg.Src
		//fmt.Println("In", localID, "<-", peerID , "set init")
	} else {
		log.Fatal(localID, " <- First Message Should be init" )
	}

	return &IncomingPeer{
		localID: localID,
		peerID: peerID,
		conn: conn,
		enc: enc,
		dec: dec,
		incoming: incoming,
		sendQueue: make(chan protocol.Message),
	}
}


func (p *IncomingPeer) handle() {
	go func(p *IncomingPeer) {
		for {
			for msg := range p.sendQueue {
				//fmt.Println("In", p.localID, "->", p.peerID, "Encode Msg From", msg.Src, "to", msg.Dst)
				err := p.enc.Encode(msg)
				if err != nil {
					log.Fatal(err)
				}
			}
		}
	}(p)

	go func(p *IncomingPeer) {
		for {
			var msg protocol.Message
			err := p.dec.Decode(&msg)
			if err != nil {
				log.Fatal(err)
			}
			if p.peerID == -1 {
				log.Fatal(p.localID, "conn should have been initialized")
			}
			//fmt.Println("In", p.localID, "<-", p.peerID, "Decode Msg From", msg.Src, "to", msg.Dst)
			p.incoming <- msg
		}
	}(p)
}
