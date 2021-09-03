package network

import (
	"fmt"
	"net"
	"log"
	"time"
    "sync"
	"github.com/bx3/goldfish/p2p/protocol"
)

type Server struct{
	addr string
	id int
	incoming chan protocol.Message
	newIncomingPeer chan *IncomingPeer
	newOutgoingPeer chan *OutgoingPeer
	control chan protocol.Signal
	goldfish *protocol.Goldfish
	outgoingPeers map[int](*OutgoingPeer)
	incomingPeers map[int](*IncomingPeer)
	IdToAddr map[int]string
	numOut int
	numIn int
    mu sync.Mutex
}

func (server *Server) listen(ln net.Listener) {
	for {
		conn, err := ln.Accept()
		//if len(server.incomingPeers) >= server.numIn {
			//conn.Close()
			//continue
		//}

		if err != nil {
			log.Fatal(err)
		}
		server.newIncomingPeer <- NewIncomingPeer(server.id, conn, server.incoming, server.control)
	}
}

//func (server *Server) lookup(addr string) int {
	//id, prs := server.addrToID[addr]
	//if !prs {
		//log.Fatal(addr, "not present in map")
	//}
	//return id
//}

func (s *Server) Connect(id int, addr string) {
	go func() {
		for {
			//if len(s.outgoingPeers) >= s.numOut {
				//log.Fatal("Cannot Add more outgoing conn, curr lim", s.numOut)
			//}

			conn, err := net.Dial("tcp", addr)
			if err != nil {
				time.Sleep(50 * time.Millisecond)
				fmt.Println(s.id, "failed to connect", id, "retry in 50 ms")
			} else {
				s.newOutgoingPeer <- NewOutgoingPeer(s.id, id, conn, s.incoming, s.control)
				break
			}
		}
	}()
}

func StartServer(id int, addr string, numOut int, numIn int, idToAddr map[int]string, goldfish *protocol.Goldfish, control chan protocol.Signal) (server *Server) {
	fmt.Println("start")
	server = &Server {
		addr: addr,
		id: id,
		incoming: make(chan protocol.Message, 40),
		newIncomingPeer: make(chan *IncomingPeer, 40),
		newOutgoingPeer: make(chan *OutgoingPeer, 40),
		control: control,
		goldfish: goldfish,
		outgoingPeers: make(map[int](*OutgoingPeer)),
		incomingPeers: make(map[int](*IncomingPeer)),
		IdToAddr: idToAddr,
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
				fmt.Println(server.id, "Managing new incoming peers", p.peerID)
                server.mu.Lock()
				server.incomingPeers[p.peerID] = p
                server.mu.Unlock()
			case p := <-server.newOutgoingPeer:
				fmt.Println(server.id, "Managing new outgoing peers", p.peerID)
				p.connect()
                server.mu.Lock()
				server.outgoingPeers[p.peerID] = p
                server.mu.Unlock()

				p.initConn()
            case signal := <-server.control:
				// Dispatch the message to outgoing peers
                if signal.Disconnected {
                    server.mu.Lock()
                    delete(server.incomingPeers, signal.DisconnectedPeerID)
                    server.mu.Unlock()
                }
                if signal.AdaptTopo {
                    server.AdaptTopo(signal.Explores, signal.Exploits)
                }
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

                server.mu.Lock()
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
                server.mu.Unlock()
			}
		}
	}()

	return server
}

func (s *Server) AdaptTopo(explores, exploits []int) {
    //is_run, explores, exploits := s.goldfish.RunGoldfishCore()
    // modify connections
    fmt.Println(s.id, "exploit", exploits, "explores", explores)

    s.mu.Lock()
    disconnects := make([]int, 0)
    for peer, _ := range s.outgoingPeers {
        isFlag := true
        for _, exploit := range exploits {
            if peer == exploit {
                isFlag = false
            }
        }
        if isFlag {
            disconnects = append(disconnects, peer)
        }
    }

    for _, peer := range exploits {
        _, prs := s.outgoingPeers[peer]
        if !prs {
            addr, addrPrs := s.IdToAddr[peer]
            if !addrPrs {
                log.Fatal(s.id, "does not know address for exploit peer", peer)
            }
            s.Connect(peer, addr)
            fmt.Println(s.id, "exploit->", peer, "addr", addr)
        }
    }

    for _, disconnect := range disconnects {
        fmt.Println(s.id, "disconnect", disconnect)
        outgoingPeer, _ := s.outgoingPeers[disconnect]
        close(outgoingPeer.sendQueue)
        delete(s.outgoingPeers, disconnect)
    }

    for _, peer := range explores {
        addr, prs := s.IdToAddr[peer]
        if !prs {
            log.Fatal(s.id, "does not know address for explore peer", peer)
        }
        fmt.Println(s.id, "explore", peer, "addr", addr)
        s.Connect(peer, addr)
    }
    s.goldfish.ResetRunning()
    s.mu.Unlock()
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

    server.mu.Lock()
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
            sent = append(sent, peer.peerID)
		}
	}
    server.mu.Unlock()
    fmt.Println(server.id, "B->", sent)

}
