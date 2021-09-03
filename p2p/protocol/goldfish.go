package protocol

import (
	"os"
    "fmt"
	"log"
	"time"
    "sync"
    "bytes"
	"bufio"
	"strconv"
	"encoding/json"
	"io/ioutil"
    "os/exec"
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

type GoldfishOutput struct {
    Explores []int `json:"explores"`
    Exploits []int `json:"exploits"`
}


type Goldfish struct {
    configJson string
    adapt bool
	CurrEpoch int
	id int
	numOut int
	numIn int
    numRand int
	numMsg int
	NumEpoch int
    UpdateInterval int
    NumTopo int
	conns []Conn
	slots [][]PeerMemory
	records map[FishMessage] int
    slotToRecord []FishMessage
	storePath string
	msgRemove int	// num msg before storing
    control chan Signal
    Running bool
    mu sync.Mutex
}

func NewGoldfish(configJson string, adapt bool, i, numOut, numIn, numRand, numMsg, numEpoch, updateInterval, numTopo int, control chan Signal) (*Goldfish) {
    storePath := "stores/node" + strconv.Itoa(i)
    os.Mkdir(storePath, 0755)
	return &Goldfish {
        configJson: configJson,
        adapt: adapt,
		CurrEpoch: 0,
		id: i,
		numOut: numOut,
		numIn: numIn,
        numRand: numRand,
		numMsg: numMsg,
		NumEpoch: numEpoch,
        UpdateInterval: updateInterval,
        NumTopo: numTopo,
		conns: make([]Conn, 0),
		slots: make([][]PeerMemory, 0),
		records: make(map[FishMessage] int),
        slotToRecord: make([]FishMessage, 0),
		storePath: storePath,
		msgRemove: 1, // last several msg are removed, they might not be complete
        control: control,
        Running: false,
	}
}

func (g *Goldfish) SetConns(conns []Conn) {
    g.mu.Lock()
	g.conns = conns
    g.mu.Unlock()
}

func (g *Goldfish) ResetRunning() {
    g.mu.Lock()
    g.Running = false
    g.mu.Unlock()
}

func (g *Goldfish) Recv(msg Message, direction Direction) []Message {

	pm := PeerMemory {
		conn: Conn{msg.Src, direction},
		t: time.Now().UnixNano(),
	}
	// store and decide what to relay
	outMsgs := []Message{}
	fm := msg.Payload

    g.mu.Lock()
	slot_i, prs := g.records[fm]
    // relay
    if !prs {
        for _, conn := range g.conns {
			if conn.Peer != msg.Src {
				newMsg := Message {
					Src: g.id,
					Dst: conn.Peer,
					IsInit: false,
					Payload : msg.Payload,
				}
				outMsgs = append(outMsgs, newMsg)
			}
		}
    }
    isRunning := g.Running
    if !isRunning {
        if !prs {
            g.records[fm] = len(g.slots)
            g.slots = append(g.slots, []PeerMemory{pm})
            g.slotToRecord = append(g.slotToRecord, fm)
        } else {
            // -1 is historic fish messages from prev topo
            if slot_i != -1 {
                g.slots[slot_i] = append(g.slots[slot_i], pm)
            }
        }
    }
    g.mu.Unlock()
    //if g.adapt {
        //fmt.Println("node", g.id, ". num slot", len(g.slots), "running", isRunning, "msg", pm)
    //}

    g.mu.Lock()
    if g.numMsg < len(g.slots) && !g.Running {
        g.Running = true
        g.mu.Unlock()
        go func() {
            is_run, explores, exploits := g.RunGoldfishCore()
            if is_run {
                signal := Signal {
                    Disconnected: false,
                    DisconnectedPeerID: 0,
                    AdaptTopo: true,
                    Explores: explores,
                    Exploits: exploits,
                }
                g.control <- signal
            } else {
                g.ResetRunning()
            }
        }()
    } else {
        g.mu.Unlock()
    }
    //if g.adapt {
        //fmt.Println("node", g.id, ". num slot", len(g.slots), "running", isRunning, "msg", pm)
    //}
	return outMsgs
}

func (g *Goldfish) RunGoldfishCore() (bool, []int, []int) {
    // record slots
    if len(g.slots) > g.msgRemove {
        if len(g.slots) > g.numMsg+g.msgRemove {
            g.recordTopo(g.slots[0:g.numMsg], g.slotToRecord[0:g.numMsg])
        } else {
            g.recordTopo(g.slots[0:len(g.slots)-g.msgRemove], g.slotToRecord[0:len(g.slots)-g.msgRemove])
        }

        g.mu.Lock()
        for fishMsg, _ := range g.records {
            g.records[fishMsg] = -1
        }
        // empty slots
        g.slots = g.slots[:0]
        g.slotToRecord = g.slotToRecord[:0]
        g.mu.Unlock()

        if !g.adapt {
            g.CurrEpoch += 1
            return false, []int{}, []int{}
        }

        if g.CurrEpoch < g.NumTopo-1 {
            fmt.Println("g.CurrEpoch", g.CurrEpoch, "g.NumTopo", g.NumTopo)
            g.CurrEpoch += 1
            return false, []int{}, []int{}
        }

        fmt.Println(g.conns)
        currOuts := make([]int, 0)
        for _, c := range g.conns {
            if c.Direct == Outgoing || c.Direct == Bidirect {
                currOuts = append(currOuts, c.Peer)
                fmt.Println(c.Peer, c.Direct.String())
            }
        }

        goldfishOutput := g.storePath + "/output" +strconv.Itoa(g.CurrEpoch)+".json"
        cmd := exec.Command(
            "python",
            "../../core/run.py",
            g.configJson,
            g.storePath,
            strconv.Itoa(g.CurrEpoch),
            strconv.Itoa(g.NumTopo),
            goldfishOutput,
            strconv.Itoa(g.numOut),
            strconv.Itoa(g.numIn),
            strconv.Itoa(g.numRand),
            strconv.Itoa(g.id))
        fmt.Println(g.id, "currOuts", currOuts, "command: ", cmd.String())
        var out bytes.Buffer
        cmd.Stdout = &out
        start := time.Now()
        err := cmd.Run()
        if err != nil {
            log.Fatal(err)
        }
        t := time.Now()
        elapsed := t.Sub(start)
        fmt.Println("Finish running Command in",  elapsed , "output",  out.String())

        // read connections
        jsonFile, err := os.Open(goldfishOutput)
        if err != nil {
            log.Fatal(err)
        }
        data, err := ioutil.ReadAll(jsonFile)
        if err != nil {
            log.Fatal(err)
        }
        output := &GoldfishOutput{}
        err = json.Unmarshal([]byte(data), output)
        if err != nil {
            log.Fatal(err)
        }

        g.CurrEpoch += 1

        if len(output.Explores)==0 && len(output.Exploits)==0 {
            fmt.Println("\t\tNo ouput", len(g.slots) , g.msgRemove)
            return false, []int{}, []int{}
        } else {
            fmt.Println("\t\tresult", output.Explores, output.Exploits)
            return true, output.Explores, output.Exploits
        }
    } else {
        print("\t\tInsufficient slots", len(g.slots) , g.msgRemove)
        return false, []int{}, []int{}
    }
}



func (g *Goldfish) recordTopo(topoSlots [][]PeerMemory, msgSlots []FishMessage) {
    topoRecords := make([][][]string, 0)
    for _, slots := range topoSlots {
        msgRecords := g.format(slots)
        topoRecords = append(topoRecords, msgRecords)
    }
    s, err := json.Marshal(topoRecords)
	if err != nil {
		log.Fatal(err)
	}

    f, err := os.Create(g.storePath + "/epoch" +strconv.Itoa(g.CurrEpoch)+"_time.json")
	if err != nil {
		log.Fatal(err)
	}
	store := bufio.NewWriter(f)

	store.Write(s)
	store.Flush()

    s, err = json.Marshal(msgSlots)
	if err != nil {
		log.Fatal(err)
	}
    f, err = os.Create(g.storePath + "/epoch" +strconv.Itoa(g.CurrEpoch)+"_msg.json")
	if err != nil {
		log.Fatal(err)
	}
	store = bufio.NewWriter(f)

	store.Write(s)
	store.Flush()
}

func (g *Goldfish) format(slots []PeerMemory) [][]string {
    records := make([][]string, 0)
	recved := make(map[int] bool)
	for _, memory := range(slots) {
		r := []string{strconv.Itoa(memory.conn.Peer), strconv.FormatInt(memory.t, 10), memory.conn.Direct.String()}
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
    return records
}

//func (g *Goldfish) record(slots []PeerMemory) {
	//records := make([][]string, 0)
	//earliest := slots[0].t
	//recved := make(map[int] bool)
	//for _, memory := range(slots) {
		//r := []string{strconv.Itoa(memory.conn.Peer), strconv.FormatInt(memory.t-earliest, 10), memory.conn.Direct.String()}
		//records = append(records, r)
		//recved[memory.conn.Peer] = true
	//}

	//for _, conn := range g.conns {
		//_, prs := recved[conn.Peer]
		//if !prs {
			//r := []string{strconv.Itoa(conn.Peer), "None", conn.Direct.String()}
			//records = append(records, r)
		//}
	//}

	//fmt.Println(records)
	//s, err := json.Marshal(records)
	//if err != nil {
		//log.Fatal(err)
	//}
	//g.store.Write(s)
	//g.store.WriteString("\n")
	//g.store.Flush()
//}

	//if len(g.slots) >= g.storeWait {
		//storeSlot := len(g.slots) - g.storeWait
		//if storeSlot == g.storeIndex {
			//g.record(g.slots[storeSlot])
			//g.storeIndex += 1
		//}
	//}


