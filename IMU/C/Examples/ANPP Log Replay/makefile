all: anpp_log_replay

anpp_log_replay: ../../Dynamic/an_packet_protocol.c ../../Dynamic/an_packet_protocol.h ../../Dynamic/ins_packets.c ../../Dynamic/ins_packets.h ../../Dynamic/rs232/rs232.c ../../Dynamic/rs232/rs232.h anpp_log_replay.c
	gcc -pedantic -Wall -o anpp_log_replay ../../Dynamic/an_packet_protocol.c ../../Dynamic/ins_packets.c ../../Dynamic/rs232/rs232.c anpp_log_replay.c
	
clean:
	rm anpp_log_replay
