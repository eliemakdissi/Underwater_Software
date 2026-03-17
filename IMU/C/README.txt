Advanced Navigation Packet Protocol Orientus SDK Version 7.3
C Language SDK

*******************
**** LIBRARIES ****
*******************

Dynamic:
	Uses dynamic memory allocation (malloc) and should be used in most cases.
	The example code connects to Orientus through the serial port and decodes data in real time.

Static:
	Does not use dynamic memory allocation and should be used on embedded systems that do not have dynamic memory allocation available.
	The example code decodes a data dump from Spatial.

******************
**** EXAMPLES ****
******************

ANPP Log Decoder:
	Contains code for decoding .anpp log files produced by Orientus Manager.

ANPP Log Replay:
	Contains code for outputting the contents of a .anpp log file over serial port.

Atmel ANPP:
	Contains code for interfacing Orientus with Atmel AVR systems.

Firmware Update:
	Code to update the device firmware via serial.

NTRIP Client:
	Code to send NTRIP correction messages to the device.

State Log Parser:
	C code for parsing Orientus Manager exported CSV state log file.

Example Logs:
	Contains a sample log file from a Spatial unit.

*******************
**** COMPILING ****
*******************

Windows:
	1. Install MinGW
	2. Open a command prompt
	3. Navigate to the directory with the makefile
	4. Type mingw32-make
	5. Type packet_example.exe
	
Linux:
	1. Open a terminal
	2. Navigate to the directory with the makefile
	3. Type make
	4. Type ./packet_example
	
*******************
**** CHANGELOG ****
*******************

v7.3:
	Added Packet 89 Vessel Motion Packet
	Added Packet 90 Automatic Magnetometer Calibration Status Packet
	Added support for Certus Mini
	Updated L band satellite IDs
v7.2:
	Added Packet 85 Sensor Temperature Packet
	Added Packet 86 System Temperature Packet
	Updated subcomponent information decoding
v7.1:
	Updated North Seeking Initialisation Status packet definition for Boreas
	Updated L band satellite IDs
v7.0:
	Replaced RS232 library
	Added ANPP Log Replay example
	Added Packet 66 GNSS Summary
	Added Packet 204 Device Name
v6.0:
	Renamed spatial_packets.h to ins_packets.h
	Renamed spatial_packets.c to ins_packets.c
	Added Packet 10 Serial Port Passthrough Packet
	Added Packet 11 IP Configuration Packet
	Added Packet 13 Extended Device Information Packet
	Added Packet 14 Subcomponent Information Packet
	Added Packet 57 Wind Packet
	Added Packet 60 Raw Satellite Data Packet
	Added Packet 61 Raw Satellite Ephemeris Packet
	Added Packet 70 Raw DVL Data Packet
	Added Packet 73 Automotive Packet
	Added Packet 75 External Magnetometers Packet
	Added Packet 80 Basestation Packet
	Added Packet 83 Zero Angular Velocity Packet
	Added Packet 84 Extended Satellites Packet
	Added Packet 85 Sensor Temperatures Packet
	Added Packet 197 GNSS Configuration Packet
	Added Packet 198 User Data Packet
	Added Packet 202 IP Dataports Configuration Packet
	Added Packet 203 CAN Configuration Packet
	Removed unsupported External Pitot Pressure Packet
	General fixes to various packet structures
	State Log Parser example uses packet definitions from ins_packets.h
v5.0:
	Added TCP socket network connection option to the Dynamic C SDK
v4.2:
	Updated UTM Position packet definition and decoding
v4.0:
	Updated GNSS receiver packet in spatial_packets.c and spatial_packets.h
	Changed NMEA configuration packet to GPIO configuration packet in spatial_packets.c and spatial_packets.h
	Updated dual antenna configuration packet in spatial_packets.c and spatial_packets.h
	Updated zero alignment packet in spatial_packets.c and spatial_packets.h
	Added odometer packet to spatial_packets.c and spatial_packets.h
	Added external air data packet to spatial_packets.c and spatial_packets.h
	Added GNSS receiver information packet to spatial_packets.c and spatial_packets.h
	RS232 library updated to use serial port path name instead of index
v3.0:
	No changes
v2.3:
	Added C++ extern declarations in header files
	Added malloc failure tolerance in dynamic example spatial_packets.c
	Added external pitot pressure packet to spatial_packets.c and spatial_packets.h
	Added wind estimation packet to spatial_packets.c and spatial_packets.h
	Added heave packet to spatial_packets.c and spatial_packets.h
	Added heave offset packet to spatial_packets.c and spatial_packets.h
v2.2:
	Removed reserved fields from typedef structs
	Added external time packet to spatial_packets.c and spatial_packets.h
	Added external depth packet to spatial_packets.c and spatial_packets.h
	Added geoid height packet to spatial_packets.c and spatial_packets.h
v2.1:
	No changes
v2.0:
	Removed packet_example.h
	Added additional acknowledge responses to spatial_packets.h
	Added running time packet to spatial_packets.c and spatial_packets.h
	Added local magnetic field packet to spatial_packets.c and spatial_packets.h
	Added odometer state packet to spatial_packets.c and spatial_packets.h
	Alignment packet modified in spatial_packets.c and spatial_packets.h
	Added additional GPIO functions to spatial_packets.h
	Added odometer configuration packet to spatial_packets.c and spatial_packets.h
	Added zero orientation alignment packet to spatial_packets.c and spatial_packets.h
v1.1:
	spatial_packets.c and spatial_packets.h now contain decode and encode functions for all current packets
v1.0:
	Added spatial_packets.c and spatial_packets.h
v0.5:
	No changes
v0.2:
	Complete overhaul
v0.1:
	First Release
