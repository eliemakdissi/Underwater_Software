/*!
 * @file main.cpp
 * @brief
 * ADC ADS1015 example file for RPI Rp2040 PICO C++ SDK
 * Shows use of single ended ADC mode.
 * URL: https://github.com/gavinlyonsrepo/ADS1x15_PICO
 */

#include "ads1x15.hpp"

PICO_ADS1015 ads;						 // 12 bit ADS1015 instance
uint16_t I2CSpeed = 100;		 // I2C speed in Khz
uint8_t DataGPIO = 4;			 // I2C GPIO for data line
uint8_t ClockGPIO = 5;			 // I2C GPIO for Clock line
uint32_t I2CTimeout = 50000; // I2C timeout delay in uS.

int16_t adc0, adc1, adc2, adc3;
float volts0, volts1, volts2, volts3;

void setup()
{
  Serial.begin(115200);
  while(!Serial) {
    delay(10);
  }
  delay(500);

	Serial.printf("ADS1x15 : Start Single End example.\r\n");
	Serial.printf("Getting Single End readings from AIN0-3");

	ads.setGain(ads.ADSXGain_ONE);

	if (!ads.beginADSX(ads.ADSX_ADDRESS_VDD, i2c0, I2CSpeed, DataGPIO, ClockGPIO, I2CTimeout))
	{
		Serial.printf("ADS1x15 : Failed to initialize ADS.!\r\n");
		while (1)
			;
	}
}

void loop() {
		adc0 = ads.readADC_SingleEnded(ads.ADSX_AIN0);
		adc1 = ads.readADC_SingleEnded(ads.ADSX_AIN1);
		adc2 = ads.readADC_SingleEnded(ads.ADSX_AIN2);
		adc3 = ads.readADC_SingleEnded(ads.ADSX_AIN3);

		volts0 = ads.computeVolts(adc0);
		volts1 = ads.computeVolts(adc1);
		volts2 = ads.computeVolts(adc2);
		volts3 = ads.computeVolts(adc3);

		Serial.printf("------------------------------\r\n");
		Serial.printf("AIN0: %i  %f V \r\n", adc0, volts0);
		Serial.printf("AIN1: %i  %f V \r\n", adc1, volts1);
		Serial.printf("AIN2: %i  %f V \r\n", adc2, volts2);
		Serial.printf("AIN3: %i  %f V \r\n", adc3, volts3);

		delay(100);
}