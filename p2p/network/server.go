package network

import (
	"fmt"
	"net"
	"log"
	"time"
	"github.com/bx3/goldfish/p2p/protocol"
	"github.com/bx3/goldfish/p2p/app"
)

type Server struct{
	addr string
	id int
	incoming chan protocol.Message
	newIncomingPeer chan *IncomingPeer
	newOutgoingPeer chan *OutgoingPeer
	scheduler chan protocol.Message
	goldfish *protocol.Goldfish
	pub *app.Publisher
	outgoingPeers map[int](*OutgoingPeer)
	incomingPeers map[int](*IncomingPeer)
	addrToID map[string]int
	numOut int
	numIn int
}

func (server *Server) listen(ln net.Listener) {
	for {
		conn, err := ln.Accept()
		if len(server.incomingPeers) >= server.numIn {
			conn.Close()
			continue
		}

		if err != nil {
			log.Fatal(err)
		}
		server.newIncomingPeer <- NewIncomingPeer(server.id, conn, server.incoming)
	}
}

func (server *Server) lookup(addr string) int {
	id, prs := server.addrToID[addr]
	if !prs {
		log.Fatal(addr, "not present in map")
	}
	return id
}

func (server *Server) Connect(id int, addr string) {
	go func() {
		for {
			if len(server.outgoingPeers) >= server.numOut {
				log.Fatal("Cannot Add more outgoing conn, curr lim", server.numOut)
			}

			conn, err := net.Dial("tcp", addr)
			if err != nil {
				time.Sleep(50 * time.Millisecond)
				fmt.Println(server.id, "failed to connect", id, "retry in 50 ms")
			} else {
				server.newOutgoingPeer <- NewOutgoingPeer(server.id, id, conn, server.incoming)
				break
			}
		}
	}()
}

func StartServer(id int, addr string, numOut int, numIn int, addrToID map[string]int, goldfish *protocol.Goldfish, pub *app.Publisher) (server *Server) {
	fmt.Println("start")
	server = &Server {
		addr: addr,
		id: id,
		incoming: make(chan protocol.Message),
		newIncomingPeer: make(chan *IncomingPeer, 40),
		newOutgoingPeer: make(chan *OutgoingPeer, 40),
		scheduler: make(chan protocol.Message),
		goldfish: goldfish,
		pub: pub,
		outgoingPeers: make(map[int](*OutgoingPeer)),
		incomingPeers: make(map[int](*IncomingPeer)),
		addrToID: addrToID,
		numOut: numOut,
		numIn: numIn,
	}

	ln, err := net.Listen("tcp", addr)
	if err != nil {
		log.Fatal(err)
	}
	go func() {
		server.listen(ln)
	}()

	go func() {
		for {
			select {
			case p := <-server.newIncomingPeer:
				p.handle()
				server.incomingPeers[p.peerID] = p
			case p := <-server.newOutgoingPeer:
				fmt.Println("Managing new peers")
				p.connect()
				server.outgoingPeers[p.peerID] = p
				p.initConn()
			case m := <-server.scheduler:
				// Dispatch the message to outgoing peers
				server.outgoingPeers[m.Dst].sendQueue <- m
			}
		}
	}()

	// processing incoming messages
	go func() {
		for m := range server.incoming {
			direction := server.GetSrcDirection(m)
			conns := server.GetConns()
			goldfish.SetConns(conns)
			outMsgs := goldfish.Recv(m, direction)
			for _, msg := range outMsgs {
				dst := msg.Dst
				out_p, prs := server.outgoingPeers[dst]
				if prs {
					out_p.sendQueue <- msg
				} else {
					in_p, prs := server.incomingPeers[dst]
					if !prs {
						log.Fatal("Unable to send msg. Dst ", dst, "not Connected")
					}
					in_p.sendQueue <- msg
				}
			}
		}
	}()

	ticker := time.NewTicker(2000 * time.Millisecond)
	done := make(chan bool)
	go func() {
		// do our own things
		time.Sleep(1000 * time.Millisecond)
		for {
			select {
			case <-ticker.C:
				ok, msg := server.pub.Publish()
				if ok {
					fmt.Println(server.id, "Publish\n")
					server.Broadcast(msg)
				}
			case <-done:
				fmt.Println("done generating source msg")
				return
			}
		}
	}()

	return server
}

func (server *Server) GetSrcDirection(msg protocol.Message) protocol.Direction {
	_, out_prs := server.outgoingPeers[msg.Src]
	_, in_prs := server.incomingPeers[msg.Src]
	if out_prs && in_prs {
		return protocol.Bidirect
	} else if out_prs {
		return protocol.Outgoing
	} else if in_prs {
		return protocol.Incoming
	} else {
		outKeys := make([]int, 0, len(server.outgoingPeers))
		for k := range server.outgoingPeers {
			outKeys = append(outKeys, k)
		}
		inKeys := make([]int, 0, len(server.incomingPeers))
		for k := range server.incomingPeers {
			inKeys = append(inKeys, k)
		}
		log.Fatal(server.id, "Unable to Get direction of msg. Src ", msg.Src, "not Connected")
		return protocol.NoConn
	}
}

func (server *Server) GetConns() []protocol.Conn {
	conns := make([]protocol.Conn, 0)
	for _, outPeer := range server.outgoingPeers {
		_, prs := server.incomingPeers[outPeer.peerID]
		if prs {
			conns = append(conns, protocol.Conn{outPeer.peerID, protocol.Bidirect})
		} else {
			conns = append(conns, protocol.Conn{outPeer.peerID, protocol.Outgoing})
		}
	}
	for _, inPeer := range server.incomingPeers {
		_, prs := server.outgoingPeers[inPeer.peerID]
		if !prs {
			conns = append(conns, protocol.Conn{inPeer.peerID, protocol.Incoming})
		}
	}
	return conns
}

func (server *Server) Broadcast(msg protocol.Message) {
	sent := make([]int, 0)

	for _, peer := range server.outgoingPeers {
		peerMsg := msg
		peerMsg.Src = server.id
		peerMsg.Dst = peer.peerID
		peer.sendQueue <- peerMsg
		sent = append(sent, peer.peerID)
	}
	for _, peer := range server.incomingPeers {
		flag := false
		for _, id := range sent {
			if id == peer.peerID {
				flag = true
			}
		}
		if !flag {
			peerMsg := msg
			peerMsg.Src = server.id
			peerMsg.Dst = peer.peerID
			peer.sendQueue <- peerMsg
		}
	}
}
