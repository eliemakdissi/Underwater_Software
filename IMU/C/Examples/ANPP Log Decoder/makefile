all: anpp_log_decoder

anpp_log_decoder: ../../Dynamic/an_packet_protocol.c ../../Dynamic/an_packet_protocol.h ../../Dynamic/ins_packets.c ../../Dynamic/ins_packets.h anpp_log_decoder.c
	gcc -pedantic -Wall -o anpp_log_decoder ../../Dynamic/an_packet_protocol.c ../../Dynamic/ins_packets.c anpp_log_decoder.c
	
clean:
	rm anpp_log_decoder
