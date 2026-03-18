all: packet_example

packet_example: an_packet_protocol.c an_packet_protocol.h spatial_data_dump.h ins_packets.c ins_packets.h packet_example.c
	gcc -pedantic -Wall -o packet_example an_packet_protocol.c ins_packets.c packet_example.c
	
clean:
	rm packet_example
