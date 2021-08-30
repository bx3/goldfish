package protocol

import (
	"os"
	"fmt"
	"log"
	"time"
	"bufio"
	"strconv"
	"encoding/json"
)

type Direction int
const (
	Incoming Direction = iota
	Outgoing
	Bidirect
	NoConn
)

func (d Direction) String() string {
	switch d {
	case Incoming:
		return "incoming"
	case Outgoing:
		return "outgoing"
	case Bidirect:
		return "bidirect"
	case NoConn:
		return "NoConn"
	default:
		return "Err"
	}
}

type FishMessage struct {
	PubID int
	PubCount int
	PubTime int64
}

type PeerMemory struct {
	conn Conn
	t int64
}

type Conn struct {
	Peer int
	Direct Direction
}


type Goldfish struct {
	CurrEpoch int
	id int
	numOut int
	numIn int
	numMsg int
	NumEpoch int
	conns []Conn
	slots [][]PeerMemory
	records map[FishMessage] int
	store *bufio.Writer
	storeWait int	// num msg before storing
	storeIndex int
}

func NewGoldfish(i, numOut, numIn, numMsg, numEpoch int) (*Goldfish) {
	f, err := os.Create("stores/"+strconv.Itoa(i)+".txt")
	if err != nil {
		log.Fatal(err)
	}
	w := bufio.NewWriter(f)
	return &Goldfish {
		CurrEpoch: 0,
		id: i,
		numOut: numOut,
		numIn: numIn,
		numMsg: numMsg,
		NumEpoch: numEpoch,
		conns: make([]Conn, 0),
		slots: make([][]PeerMemory, 0),
		records: make(map[FishMessage] int),
		store: w,
		storeWait: 2,
		storeIndex: 0,
	}
}

func (g *Goldfish) SetConns(conns []Conn) {
	g.conns = conns
}

func (g *Goldfish) Recv(msg Message, direction Direction) []Message {
	fmt.Println("Goldfish", g.id, "<-", msg.Src, "conns", g.conns)
	pm := PeerMemory {
		conn: Conn{msg.Src, direction},
		t: time.Now().UnixNano(),
	}
	// process
	outMsgs := []Message{}
	fm := msg.Payload
	slot_i, prs := g.records[fm]
	if !prs {
		g.records[fm] = len(g.slots)
		g.slots = append(g.slots, []PeerMemory{pm})
		// relay
		for _, conn := range g.conns {
			if conn.Peer != msg.Src {
				newMsg := Message {
					Src: g.id,
					Dst: conn.Peer,
					IsInit: false,
					Payload : msg.Payload,
				}
				fmt.Println("Goldfish", g.id, "relay ->", conn.Peer)
				outMsgs = append(outMsgs, newMsg)
			}
		}
	} else {
		g.slots[slot_i] = append(g.slots[slot_i], pm)
	}

	if len(g.slots) >= g.storeWait {
		storeSlot := len(g.slots) - g.storeWait
		if storeSlot == g.storeIndex {
			g.record(g.slots[storeSlot])
			g.storeIndex += 1
		}
	}

	return outMsgs
}

func (g *Goldfish) record(slots []PeerMemory) {
	records := make([][]string, 0)
	earliest := slots[0].t
	recved := make(map[int] bool)
	for _, memory := range(slots) {
		r := []string{strconv.Itoa(memory.conn.Peer), strconv.FormatInt(memory.t-earliest, 10), memory.conn.Direct.String()}
		records = append(records, r)
		recved[memory.conn.Peer] = true
	}

	for _, conn := range g.conns {
		_, prs := recved[conn.Peer]
		if !prs {
			r := []string{strconv.Itoa(conn.Peer), "None", conn.Direct.String()}
			records = append(records, r)
		}
	}

	fmt.Println(records)
	s, err := json.Marshal(records)
	if err != nil {
		log.Fatal(err)
	}
	g.store.Write(s)
	g.store.WriteString("\n")
	g.store.Flush()
}


