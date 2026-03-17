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
#define _USE_MATH_DEFINES
#include <math.h>

#ifdef _WIN32
#include <Windows.h>
#else
#include <unistd.h>
#endif

#include "NTRIP/ntripclient.h"
#include "../../Dynamic/rs232/rs232.h"
#include "../../Dynamic/an_packet_protocol.h"
#include "../../Dynamic/ins_packets.h"

#define RADIANS_TO_DEGREES (180.0/M_PI)

int comPortIndex = -1;

int an_packet_transmit(an_packet_t* an_packet)
{
	an_packet_encode(an_packet);
	return comWrite(comPortIndex, an_packet_pointer(an_packet), an_packet_size(an_packet));
}

int main(int argc, char** argv)
{
	struct Args args;

	an_decoder_t an_decoder;
	an_packet_t* an_packet;
	system_state_packet_t system_state_packet;

	char buf[MAXDATASIZE];
	int error = 0;
	int bytes_received;
	int numbytes = 0;
	int remain = numbytes;
	int pos = 0;
	getargs(argc, argv, &args);

	/* Find the serial port */
	comEnumerate();
	comPortIndex = comFindPort(args.serdevice);
	if(comPortIndex == -1)
	{
		printf("Serial port not available\n");
		exit(EXIT_FAILURE);
	}
	/* Open the serial port */
	if(comOpen(comPortIndex, args.baud) == 0)
	{
		printf("Could not open serial port\n");
		exit(EXIT_FAILURE);
	}

  	an_decoder_initialise(&an_decoder);

	error = ntrip_initialise(&args, buf);

	while (!error)
	{
		if ((bytes_received = comRead(comPortIndex, an_decoder_pointer(&an_decoder), an_decoder_size(&an_decoder))) > 0)
		{
			/* increment the decode buffer length by the number of bytes received */
			an_decoder_increment(&an_decoder, bytes_received);

			/* decode all the packets in the buffer */
			while ((an_packet = an_packet_decode(&an_decoder)) != NULL)
			{
				if (an_packet->id == packet_id_system_state) /* system state packet */
				{
					/* copy all the binary data into the typedef struct for the packet */
					/* this allows easy access to all the different values             */
					if(decode_system_state_packet(&system_state_packet, an_packet) == 0)
					{
						printf("System State Packet:\n");
						printf("\tLatitude = %f, Longitude = %f, Height = %f\n", system_state_packet.latitude * RADIANS_TO_DEGREES, system_state_packet.longitude * RADIANS_TO_DEGREES, system_state_packet.height);
						printf("\tRoll = %f, Pitch = %f, Heading = %f\n", system_state_packet.orientation[0] * RADIANS_TO_DEGREES, system_state_packet.orientation[1] * RADIANS_TO_DEGREES, system_state_packet.orientation[2] * RADIANS_TO_DEGREES);
						printf("Fix Type:\t");

						switch(system_state_packet.filter_status.b.gnss_fix_type)
						{
							case 0:
								printf("No GNSS fix\n");
								break;
							case 1:
								printf("2D Fix\n");
								break;
							case 2:
								printf("3D Fix\n");
								break;
							case 3:
								printf("SBAS Fix\n");
								break;
							case 4:
								printf("Differential Fix\n");
								break;
							case 5:
								printf("Omnistar/Starfire Fix\n");
								break;
							case 6:
								printf("RTK Float\n");
								break;
							case 7:
								printf("RTK Fixed\n");
								break;
							default:
								printf("Unknown\n");
						}
					}
				}
				else
				{
					printf("Packet ID %u of Length %u\n", an_packet->id, an_packet->length);
				}

				/* Ensure that you free the an_packet when your done with it or you will leak memory */
				an_packet_free(&an_packet);
			}
		}

		//Get RTCM data from server
		error = ntrip(&args, buf, &numbytes);
		remain = numbytes;
		//Send Buffer in 255 Byte chunks to the device
		while(remain)
		{
			uint8_t toCpy = remain > AN_MAXIMUM_PACKET_SIZE ? AN_MAXIMUM_PACKET_SIZE : (uint8_t)remain;
			rtcm_corrections_packet_t rtcm_packet;
			rtcm_packet.packet_data = (unsigned char*) buf+pos;
			an_packet = encode_rtcm_corrections_packet(&rtcm_packet, toCpy);
			an_packet_transmit(an_packet);
			an_packet_free(&an_packet);
			pos += toCpy;			// Increment buffer
			remain -= toCpy;
		}

		pos=0;

#ifdef _WIN32
		Sleep(10);
#else
		usleep(10000);
#endif

	}

	return(0);
}
