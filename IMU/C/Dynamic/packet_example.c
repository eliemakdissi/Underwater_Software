/******************************************************************************/
/*                                                                            */
/*                Advanced Navigation Packet Protocol Library                 */
/*                C Language Dynamic Orientus SDK, Version 7.3                */
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

#define RS232 1
#define NETWORK 0

#define CONNECTION_TYPE RS232

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include <time.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>
#if CONNECTION_TYPE == NETWORK
#define _WIN32_WINNT 0x0501
#include <winsock2.h>
#include <ws2tcpip.h>
#endif
#else
#include <unistd.h>
#if CONNECTION_TYPE == RS232
#include "rs232/rs232.h"
#elif CONNECTION_TYPE == NETWORK
#include <sys/socket.h>
#include <arpa/inet.h>
#include <netdb.h>
#endif
#endif

#include "an_packet_protocol.h"
#include "ins_packets.h"

#define RADIANS_TO_DEGREES (180.0 / M_PI)

static unsigned char request_all_configuration[] = {0xE2, 0x01, 0x10, 0x9A, 0x73, 0xB6, 0xB4, 0xB5, 0xB8, 0xB9, 0xBA, 0xBC, 0xBD, 0xC0, 0xC2, 0xC3, 0xC4, 0x03, 0xC6, 0x45, 0xC7};
int comPortIndex = -1;
int socket_fd = -1;

int transmit(const unsigned char *data, int length);
int receive(unsigned char *data, int length);
int an_packet_transmit(an_packet_t *an_packet);
void set_filter_options();
void send_sensor_ranges_configuration(sensor_ranges_packet_t *ranges_input);
void send_baud_rates_configuration(baud_rates_packet_t *baud_rates);
void send_packet_periods_packet(packet_periods_packet_t *periods);
void send_packet_timer_period_packet(packet_timer_period_packet_t *master_timer);

int main(int argc, char *argv[])
{
	an_decoder_t an_decoder;
	an_packet_t *an_packet;

	system_state_packet_t system_state_packet;
	raw_sensors_packet_t raw_sensors_packet;
	sensor_ranges_packet_t sensor_ranges_packet;
	baud_rates_packet_t baud_rates_packet;
	packet_periods_packet_t packet_periods_packet;
	packet_timer_period_packet_t packet_timer_period_packet;

	FILE *anpp_log_file;

	char filename[64];
	time_t rawtime;
	struct tm *timeinfo;
	int write_counter = 0;
	int bytes_received;

#if CONNECTION_TYPE == RS232
	if (argc != 3)
	{
		printf("Incorrect number of arguments supplied\n");
		printf("Usage - %s com_port baud_rate\n", argv[0]);
#ifdef _WIN32
		printf("Example - %s COM1 115200\n", argv[0]);
#else
		printf("Example - %s ttyUSB0 115200\n", argv[0]);
#endif
		exit(EXIT_FAILURE);
	}

	/* Find the serial port */
	comEnumerate();
	comPortIndex = comFindPort(argv[1]);
	if (comPortIndex == -1)
	{
		printf("Serial port not available\n");
		exit(EXIT_FAILURE);
	}
	/* Open the serial port */
	if (comOpen(comPortIndex, atoi(argv[2])) == 0)
	{
		printf("Could not open serial port\n");
		exit(EXIT_FAILURE);
	}

#elif CONNECTION_TYPE == NETWORK
	if (argc != 3)
	{
		printf("Incorrect number of arguments supplied\n");
		printf("Usage - %s ip_address tcp_port\n", argv[0]);
		printf("Example - %s 192.168.1.54 16718\n", argv[0]);
		exit(EXIT_FAILURE);
	}

#if _WIN32
	WSADATA wsa;
	if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0)
	{
		printf("Failed. Error Code : %d", WSAGetLastError());
		exit(EXIT_FAILURE);
	}
#endif

	struct addrinfo hints = {0}, *addresses;
	hints.ai_family = AF_INET;
	hints.ai_socktype = SOCK_STREAM;
	hints.ai_protocol = IPPROTO_TCP;
	if (getaddrinfo(argv[1], argv[2], &hints, &addresses))
	{
		printf("Failure resolving hostname %s, port %s\n", argv[1], argv[2]);
		exit(EXIT_FAILURE);
	}

	if ((socket_fd = socket(addresses->ai_family, addresses->ai_socktype, addresses->ai_protocol)) < 0)
	{
		printf("Socket creation error \n");
		freeaddrinfo(addresses);
		exit(EXIT_FAILURE);
	}

	if (connect(socket_fd, addresses->ai_addr, addresses->ai_addrlen) < 0)
	{
		printf("Socket connection failed \n");
		freeaddrinfo(addresses);
		exit(EXIT_FAILURE);
	}
	freeaddrinfo(addresses);
#else
	printf("Invalid CONNECTION_TYPE\n");
	exit(EXIT_FAILURE);
#endif

	/* Create a Log file */
	rawtime = time(NULL);
	timeinfo = localtime(&rawtime);
	sprintf(filename, "ANLog_%02d-%02d-%02d_%02d-%02d-%02d.anpp", timeinfo->tm_year - 100, timeinfo->tm_mon + 1, timeinfo->tm_mday, timeinfo->tm_hour, timeinfo->tm_min, timeinfo->tm_sec);
	anpp_log_file = fopen(filename, "wb");

	sensor_ranges_packet_t my_config;
	my_config.permanent = 0;
	my_config.accelerometers_range = 2; // 16G
	my_config.gyroscopes_range = 2;		// 2000 dps
	my_config.magnetometers_range = 2;	// 8 Gauss
	send_sensor_ranges_configuration(&my_config);

	baud_rates_packet_t new_baud_config;
	new_baud_config.permanent = 1;
	new_baud_config.primary_baud_rate = 2000000;
	new_baud_config.gpio_1_2_baud_rate = 115200;
	new_baud_config.auxiliary_baud_rate = 115200;
	new_baud_config.reserved = 0;
	send_baud_rates_configuration(&new_baud_config);
	usleep(20000);
	comClose(comPortIndex);
	if (comOpen(comPortIndex, new_baud_config.primary_baud_rate))
	{
		printf("Successfully re-opened port at %d baud!\n", new_baud_config.primary_baud_rate);
	}
	else
	{
		printf("Failed to open port at high speed. Check your USB adapter capabilities.\n");
		return 1;
	}

	// packet rate (Hz) = 1000000/[(packet period)*(packet master timer period)]
	packet_timer_period_packet_t master_timer;
	master_timer.permanent = 1;
	master_timer.utc_synchronisation = 0;
	master_timer.packet_timer_period = 1000;
	send_packet_timer_period_packet(&master_timer);

	packet_periods_packet_t periods;
	periods.permanent = 1;
	periods.clear_existing_packets = 1;
	periods.packet_periods[0].packet_id = 20;
	periods.packet_periods[0].period = 1;
	periods.packet_periods[1].packet_id = 28;
	periods.packet_periods[1].period = 1;
	// Set the rest to ID 0 so the sensor ignores these slots
	for (int i = 2; i < 5; i++)
	{
		periods.packet_periods[i].packet_id = 0;
		periods.packet_periods[i].period = 0;
	}
	send_packet_periods_packet(&periods);

	/* Request all the configuration and the device information from the unit */
	transmit(request_all_configuration, sizeof(request_all_configuration));

	an_decoder_initialise(&an_decoder);

	while (1)
	{
		if ((bytes_received = receive(an_decoder_pointer(&an_decoder), an_decoder_size(&an_decoder))) > 0)
		{
			/* Log all binary data coming from the sensor, this can be converted to CSV using the manager software */
			fwrite(an_decoder_pointer(&an_decoder), sizeof(uint8_t), bytes_received, anpp_log_file);
			if (write_counter++ >= 100)
			{
				fflush(anpp_log_file);
				write_counter = 0;
			}

			/* increment the decode buffer length by the number of bytes received */
			an_decoder_increment(&an_decoder, bytes_received);

			/* decode all the packets in the buffer */
			while ((an_packet = an_packet_decode(&an_decoder)) != NULL)
			{
				if (an_packet->id == packet_id_baud_rates)
				{
					if (decode_baud_rates_packet(&baud_rates_packet, an_packet) == 0)
					{
						printf("\n--- BAUD RATE CONFIGURATION RECEIVED ---\n");
						printf("Permanent Save: %s\n", baud_rates_packet.permanent ? "Yes" : "No");

						printf("Primary Baud:   %u bps\n", baud_rates_packet.primary_baud_rate);

						printf("GPIO 1 & 2:     %u bps\n", baud_rates_packet.gpio_1_2_baud_rate);
						printf("Auxiliary:      %u bps\n", baud_rates_packet.auxiliary_baud_rate);

						printf("------------------------------\n");
					}
				}
				else if (an_packet->id == packet_id_packet_timer_period)
				{
					if (decode_packet_timer_period_packet(&packet_timer_period_packet, an_packet) == 0)
					{
						printf("\n--- MASTER TIMER CONFIGURATION RECEIVED ---\n");
						printf("Permanent Save: %s\n", packet_timer_period_packet.permanent ? "Yes" : "No");

						uint16_t period_us = packet_timer_period_packet.packet_timer_period;
						printf("Base Period:    %u us\n", period_us);

						if (period_us > 0)
						{
							float base_hz = 1000000.0f / (float)period_us;
							printf("Base Frequency: %.1f Hz\n", base_hz);
						}

						printf("------------------------------\n");
					}
				}
				else if (an_packet->id == packet_id_packet_periods)
				{
					if (decode_packet_periods_packet(&packet_periods_packet, an_packet) == 0)
					{
						printf("\n--- ACTIVE PACKET PERIODS CONFIGURATION RECEIVED ---\n");
						printf("Permanent Save: %s\n", packet_periods_packet.permanent ? "Yes" : "No");
						printf("Clear Existing: %s\n", packet_periods_packet.clear_existing_packets ? "Yes" : "No");

						int active_count = 0;
						for (int i = 0; i < 5; i++)
						{
							uint8_t id = packet_periods_packet.packet_periods[i].packet_id;

							if (id != 0)
							{
								uint32_t period = packet_periods_packet.packet_periods[i].period;

								// Frequency calculation (1,000,000 / period in microseconds)
								float frequency = (period > 0) ? (1000000.0f / (float)period) : 0.0f;

								printf("  -> Packet ID %3d: %7u us (%6.1f Hz)\n", id, period, frequency);
								active_count++;
							}
						}

						if (active_count == 0)
						{
							printf("  (No packets are currently set to stream)\n");
						}
						printf("------------------------------\n");
					}
				}
				else if (an_packet->id == packet_id_sensor_ranges)
				{
					if (decode_sensor_ranges_packet(&sensor_ranges_packet, an_packet) == 0)
					{

						printf("\n--- SENSOR RANGES CONFIGURATION RECEIVED ---\n");
						printf("Permanent Save: %s\n", sensor_ranges_packet.permanent ? "Yes" : "No");

						printf("Accel Range:    %d (%s)\n", sensor_ranges_packet.accelerometers_range,
							   (sensor_ranges_packet.accelerometers_range == 0) ? "2G" : (sensor_ranges_packet.accelerometers_range == 1) ? "4G"
																																		  : "16G");

						printf("Gyro Range:     %d (%s)\n", sensor_ranges_packet.gyroscopes_range,
							   (sensor_ranges_packet.gyroscopes_range == 0) ? "250 dps" : (sensor_ranges_packet.gyroscopes_range == 1) ? "500 dps"
																																	   : "2000 dps");

						printf("Mag Range:      %d (%s)\n", sensor_ranges_packet.magnetometers_range,
							   (sensor_ranges_packet.magnetometers_range == 0) ? "2 Gauss" : (sensor_ranges_packet.magnetometers_range == 1) ? "4 Gauss"
																																			 : "8 Gauss");

						printf("------------------------------\n");
					}
				}
				else if (an_packet->id == packet_id_system_state) /* system state packet */
				{
					/* copy all the binary data into the typedef struct for the packet */
					/* this allows easy access to all the different values             */
					if (decode_system_state_packet(&system_state_packet, an_packet) == 0)
					{
						printf("System State Packet:\n");
						printf("\tLatitude = %f, Longitude = %f, Height = %f\n", system_state_packet.latitude * RADIANS_TO_DEGREES, system_state_packet.longitude * RADIANS_TO_DEGREES, system_state_packet.height);
						printf("\tRoll = %f, Pitch = %f, Heading = %f\n", system_state_packet.orientation[0] * RADIANS_TO_DEGREES, system_state_packet.orientation[1] * RADIANS_TO_DEGREES, system_state_packet.orientation[2] * RADIANS_TO_DEGREES);
					}
				}
				else if (an_packet->id == packet_id_raw_sensors) /* raw sensors packet */
				{
					/* copy all the binary data into the typedef struct for the packet */
					/* this allows easy access to all the different values             */
					if (decode_raw_sensors_packet(&raw_sensors_packet, an_packet) == 0)
					{
						printf("Raw Sensors Packet:\n");
						printf("\tAccelerometers X: %f Y: %f Z: %f\n", raw_sensors_packet.accelerometers[0], raw_sensors_packet.accelerometers[1], raw_sensors_packet.accelerometers[2]);
						printf("\tGyroscopes X: %f Y: %f Z: %f\n", raw_sensors_packet.gyroscopes[0] * RADIANS_TO_DEGREES, raw_sensors_packet.gyroscopes[1] * RADIANS_TO_DEGREES, raw_sensors_packet.gyroscopes[2] * RADIANS_TO_DEGREES);
						printf("\tMagnetometers X: %f Y: %f Z: %f\n", raw_sensors_packet.magnetometers[0], raw_sensors_packet.magnetometers[1], raw_sensors_packet.magnetometers[2]);
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
#ifdef _WIN32
		Sleep(10);
#else
		usleep(10000);
#endif
	}
}

int transmit(const unsigned char *data, int length)
{
#if CONNECTION_TYPE == RS232
	return comWrite(comPortIndex, data, length);
#elif CONNECTION_TYPE == NETWORK
#if _WIN32
	return send(socket_fd, (char *)data, length, 0);
#else
	return write(socket_fd, data, length);
#endif
#endif
}

int receive(unsigned char *data, int length)
{
#if CONNECTION_TYPE == RS232
	return comRead(comPortIndex, data, length);
#elif CONNECTION_TYPE == NETWORK
#if _WIN32
	return recv(socket_fd, (char *)data, length, 0);
#else
	return read(socket_fd, data, length);
#endif
#endif
}

int an_packet_transmit(an_packet_t *an_packet)
{
	an_packet_encode(an_packet);
	return transmit(an_packet_pointer(an_packet), an_packet_size(an_packet));
}

/*
 * This is an example of sending a configuration packet to Orientus.
 *
 * 1. First declare the structure for the packet, in this case filter_options_packet_t.
 * 2. Set all the fields of the packet structure
 * 3. Encode the packet structure into an an_packet_t using the appropriate helper function
 * 4. Send the packet
 * 5. Free the packet
 */
void set_filter_options()
{
	an_packet_t *an_packet;
	filter_options_packet_t filter_options_packet;

	/* initialise the structure by setting all the fields to zero */
	memset(&filter_options_packet, 0, sizeof(filter_options_packet_t));

	filter_options_packet.permanent = TRUE;
	filter_options_packet.vehicle_type = vehicle_type_car;
	filter_options_packet.magnetometers_enabled = TRUE;

	an_packet = encode_filter_options_packet(&filter_options_packet);

	an_packet_transmit(an_packet);

	an_packet_free(&an_packet);
}

void send_sensor_ranges_configuration(sensor_ranges_packet_t *ranges_input)
{
	an_packet_t *an_packet = encode_sensor_ranges_packet(ranges_input);

	if (an_packet != NULL)
	{
		an_packet_transmit(an_packet);
		an_packet_free(&an_packet);
		printf("Sent Range Configuration!");
	}
}

void send_baud_rates_configuration(baud_rates_packet_t *baud_rates)
{
	an_packet_t *an_packet = encode_baud_rates_packet(baud_rates);

	if (an_packet != NULL)
	{
		an_packet_transmit(an_packet);
		an_packet_free(&an_packet);
		printf("Sent Baud Rates!");
	}
}

void send_packet_periods_packet(packet_periods_packet_t *periods)
{
	an_packet_t *an_packet = encode_packet_periods_packet(periods);

	if (an_packet != NULL)
	{
		an_packet_transmit(an_packet);
		an_packet_free(&an_packet);
		printf("Sent Periods!");
	}
}

void send_packet_timer_period_packet(packet_timer_period_packet_t *master_timer)
{
	an_packet_t *an_packet = encode_packet_timer_period_packet(master_timer);

	if (an_packet != NULL)
	{
		an_packet_transmit(an_packet);
		an_packet_free(&an_packet);
		printf("Sent Master Timer!");
	}
}