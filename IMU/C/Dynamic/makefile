LIBRARIES=
ifeq ($(OS),Windows_NT)
	LIBRARIES += -lWs2_32
endif

all: packet_example

packet_example: rs232/rs232.c rs232/rs232.h an_packet_protocol.c an_packet_protocol.h ins_packets.c ins_packets.h packet_example.c
	gcc -pedantic -Wall -o packet_example rs232/rs232.c an_packet_protocol.c ins_packets.c packet_example.c $(LIBRARIES)
	
clean:
	rm packet_example
