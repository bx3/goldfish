package protocol

type Message struct {
	Dst int
	Src int
	IsInit bool
	Payload FishMessage
}
