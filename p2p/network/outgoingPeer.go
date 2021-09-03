package network

import (
    "fmt"
	"net"
	"encoding/gob"
	"github.com/bx3/goldfish/p2p/protocol"
	"log"
)

type OutgoingPeer struct{
	localID int
	peerID int
	conn net.Conn
	enc *gob.Encoder
	dec *gob.Decoder
	incoming chan protocol.Message
	sendQueue chan protocol.Message
    control chan protocol.Signal
}

func NewOutgoingPeer(id int, peerID int, conn net.Conn, incoming chan protocol.Message, control chan protocol.Signal) *OutgoingPeer {
	enc := gob.NewEncoder(conn)
	dec := gob.NewDecoder(conn)
	return &OutgoingPeer {
		localID: id,
		peerID: peerID,
		conn: conn,
		enc: enc,
		dec: dec,
		incoming: incoming,
		sendQueue: make(chan protocol.Message),
        control: control,
	}
}

func (p *OutgoingPeer) initConn() {
	initMsg := protocol.Message{}
	initMsg.IsInit = true
	initMsg.Src = p.localID
	//fmt.Println(p.localID, "->", p.peerID, "init")
	p.sendQueue <- initMsg
}

func (p *OutgoingPeer) disconnect() {
    p.conn.Close()
}

func (p *OutgoingPeer) connect() {
	go func(p *OutgoingPeer) {
        for msg := range p.sendQueue {
            err := p.enc.Encode(msg)
            if err != nil {
                log.Fatal(err)
            }
        }
        p.conn.Close()
	}(p)

	go func(p *OutgoingPeer) {
		for {
			var msg protocol.Message
			err := p.dec.Decode(&msg)
			if err != nil {
                fmt.Println(p.localID, "<-", p.peerID, " outgoing sender ends close")
                break
			}

			if p.peerID == -1 {
				log.Fatal(p.localID, "conn should have been initialized")
			}

			p.incoming <- msg
		}
	}(p)
}
