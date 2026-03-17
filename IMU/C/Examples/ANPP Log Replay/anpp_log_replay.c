/******************************************************************************/
/*                                                                            */
/*                Advanced Navigation Packet Protocol Library                 */
/*                    C Language Orientus SDK, Version 7.3                    */
/*                    Copyright 2024, Advanced Navigation                     */
/*                                                                            */
/******************************************************************************/
/*
 * Copyright (C) 2024 Advanced Navigation
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <sys/time.h>
#define _USE_MATH_DEFINES
#include <math.h>

#ifdef _WIN32
#include <Windows.h>
#else
#include <unistd.h>
#endif

#include "../../Dynamic/rs232/rs232.h"
#include "../../Dynamic/an_packet_protocol.h"
#include "../../Dynamic/ins_packets.h"

int comPortIndex = -1;

int an_packet_transmit(an_packet_t* an_packet)
{
	an_packet_encode(an_packet);
	return comWrite(comPortIndex, an_packet_pointer(an_packet), an_packet_size(an_packet));
}

uint64_t current_time_millis()
{
	struct timeval time;
	gettimeofday(&time, NULL);
	uint64_t s1 = (uint64_t)(time.tv_sec) * 1000;
	uint64_t s2 = (time.tv_usec / 1000);
	return s1 + s2;
}

void sleep_milliseconds(uint32_t milliseconds)
{
#ifdef _WIN32
	Sleep(milliseconds);
#else
	usleep(milliseconds * 1000);
#endif
}

int main(int argc, char* argv[])
{
	an_decoder_t an_decoder;
	an_packet_t* an_packet;
	
	system_state_packet_t system_state_packet;
	unix_time_packet_t unix_time_packet;

	uint64_t start_time_ms;
	uint64_t start_packet_time_ms;
	uint64_t last_packet_time_ms;
	
	FILE* log_file;

	int bytes_read;

	if (argc != 4)
	{
		printf("Incorrect number of arguments supplied\n");
		printf("Usage - %s log_file.anpp com_port baud_rate\n", argv[0]);
#ifdef _WIN32
		printf("Example - %s ANLog.anpp COM1 115200\n", argv[0]);
#else
		printf("Example - %s ANLog.anpp ttyUSB0 115200\n", argv[0]);
#endif
		exit(EXIT_FAILURE);
	}

	log_file = fopen(argv[1], "rb");
	
	if (log_file == NULL)
	{
		printf("Error opening log file\n");
		exit(EXIT_FAILURE);
	}

	/* Find the serial port */
	comEnumerate();
	comPortIndex = comFindPort(argv[2]);
	if(comPortIndex == -1)
	{
		printf("Serial port not available\n");
		exit(EXIT_FAILURE);
	}
	/* Open the serial port */
	if(comOpen(comPortIndex, atoi(argv[3])) == 0)
	{
		printf("Could not open serial port\n");
		exit(EXIT_FAILURE);
	}
	
	an_decoder_initialise(&an_decoder);

	start_time_ms = current_time_millis();
	start_packet_time_ms = 0;
	last_packet_time_ms = 0;

	while ((bytes_read = fread(an_decoder_pointer(&an_decoder), sizeof(uint8_t), an_decoder_size(&an_decoder), log_file)) > 0)
	{
		/* Increment the decode buffer length by the number of bytes received */
		an_decoder_increment(&an_decoder, bytes_read);

		/* Decode all the packets in the buffer */
		while ((an_packet = an_packet_decode(&an_decoder)) != NULL)
		{
			if (an_packet->id == packet_id_system_state) /* system state packet */
			{
				if (decode_system_state_packet(&system_state_packet, an_packet) == 0)
				{
					last_packet_time_ms = system_state_packet.unix_time_seconds * 1000 + system_state_packet.microseconds / 1000;
				}
			}
			else if (an_packet->id == packet_id_unix_time) /* unix time packet */
			{
				if (decode_unix_time_packet(&unix_time_packet, an_packet) == 0)
				{
					last_packet_time_ms = unix_time_packet.unix_time_seconds * 1000 + unix_time_packet.microseconds / 1000;
				}
			}
			if (start_packet_time_ms == 0)
			{
				start_packet_time_ms = last_packet_time_ms;
			}

			/* If we have a valid timestamp, wait until enough time has passed since starting the replay before sending the packet */
			while ((last_packet_time_ms > 0) && ((current_time_millis() - start_time_ms) < (last_packet_time_ms - start_packet_time_ms)))
			{
				sleep_milliseconds(1);
			}

			printf("Packet ID %u of Length %u\n", an_packet->id, an_packet->length);

			/* Send the packet to the serial port */
			an_packet_transmit(an_packet);

			/* Ensure that you free the an_packet when your done with it or you will leak memory */
			an_packet_free(&an_packet);
		}
	}
	return EXIT_SUCCESS;
}
