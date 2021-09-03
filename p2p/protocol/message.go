package protocol

type Message struct {
	Dst int
	Src int
	IsInit bool
	Payload FishMessage
}

type Signal struct {
    Disconnected bool
    DisconnectedPeerID int
    AdaptTopo bool
    Explores []int
    Exploits []int
}
