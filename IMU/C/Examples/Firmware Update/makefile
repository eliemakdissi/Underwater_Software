all: firmware_update

firmware_update: ../../Dynamic/an_packet_protocol.c ../../Dynamic/an_packet_protocol.h ../../Dynamic/ins_packets.c ../../Dynamic/ins_packets.h ../../Dynamic/rs232/rs232.c ../../Dynamic/rs232/rs232.h firmware_update.c
	gcc -pedantic -Wall -o firmware_update ../../Dynamic/an_packet_protocol.c ../../Dynamic/ins_packets.c ../../Dynamic/rs232/rs232.c firmware_update.c
	
clean:
	rm firmware_update
