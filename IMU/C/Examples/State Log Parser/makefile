all: state_log_parser

state_log_parser: ../../Dynamic/an_packet_protocol.c ../../Dynamic/an_packet_protocol.h ../../Dynamic/ins_packets.c ../../Dynamic/ins_packets.h state_log_parser.c
	gcc -pedantic -Wall -o state_log_parser ../../Dynamic/an_packet_protocol.c ../../Dynamic/ins_packets.c state_log_parser.c
	
clean:
	rm state_log_parser
