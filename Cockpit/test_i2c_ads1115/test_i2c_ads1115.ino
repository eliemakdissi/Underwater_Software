#include "hardware/i2c.h"

void setup() {
    // Enable UART so we can print status output
    Serial.begin(115200);
    while(!Serial) {
      delay(10);
    }

    Serial.println("hello world");
    // This example will use I2C0 on the default SDA and SCL pins (GP4, GP5 on a Pico)
    i2c_init(i2c_default, 100 * 1000);
    gpio_set_function(PICO_DEFAULT_I2C_SDA_PIN, GPIO_FUNC_I2C);
    gpio_set_function(PICO_DEFAULT_I2C_SCL_PIN, GPIO_FUNC_I2C);
    gpio_pull_up(PICO_DEFAULT_I2C_SDA_PIN);
    gpio_pull_up(PICO_DEFAULT_I2C_SCL_PIN);

    delay(100);
    
    Serial.println("init : ");

    test1();
    test2();
    test3();
    test4();
    //int status = i2c_read_blocking(i2c_default, 0x00, );
}

void loop() {
    delay(50);
}

//simple acknowledge
void test1() {
  uint8_t msg = 0;
  i2c_read_blocking(i2c_default, 0x48, &msg, 1, false);
  Serial.println(msg == 0 ? "connneted" : "no device at 0x48");
}

//read configuration registers
uint16_t readRegister(const uint8_t reg) {
  //ask for specific register
  i2c_write_blocking (i2c_default, 0x48, &reg, 1, false);

  //read 2 bytes
  uint8_t msg[2];
  i2c_read_blocking(i2c_default, 0x48, &msg[0], 2, false);

  return msg[0] << 8 | msg[1];
}

void test2() {
  uint16_t msg = readRegister(0b10);
  Serial.print("configuration register : ");
  Serial.println(msg, BIN);
}


//write configuration (and check change)
void test3() {
  Serial.print("before : ");
  Serial.println(readRegister(0b10), BIN);

  //write bytes (default : 0b01, 0b00000101, 0b10000011)
  uint8_t data[3] = {0b01, 0b00000101, 0b10000011};
  i2c_write_blocking(i2c_default, 0x48, &data[0], 3, false);

  Serial.print("after : ");
  Serial.println(readRegister(0b10), BIN);
}

//read conversion value
void test4() {


  uint16_t value = readRegister(0b01);
  Serial.print("reading register : ");
  Serial.println(value, BIN);
  Serial.print("voltage : ");
  Serial.println(value * 2.048 / 32767.0);
}

