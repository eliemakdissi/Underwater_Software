all: ntrip_example

ifdef windir
LIBS = -lwsock32
endif

ntrip_example: ../../Dynamic/an_packet_protocol.c ../../Dynamic/an_packet_protocol.h ../../Dynamic/ins_packets.c ../../Dynamic/ins_packets.h ../../Dynamic/rs232/rs232.c ../../Dynamic/rs232/rs232.h NTRIP/ntripclient.c NTRIP/ntripclient.h ntrip_example.c
	gcc -pedantic -Wall -o ntrip_example ../../Dynamic/ins_packets.c ../../Dynamic/an_packet_protocol.c ../../Dynamic/rs232/rs232.c NTRIP/ntripclient.c ntrip_example.c $(LIBS) 

clean:
	rm ntrip_example core*
