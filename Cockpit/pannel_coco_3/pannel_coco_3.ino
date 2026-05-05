#include <BleGamepad.h>

//==== POTENTIOMETERS ====//
#define A0 7
#define A1 8
#define A2 9
#define NB_SAMPLE 32

const float deadzone = 0.0;
const float range = 1.0;
const int a0_min = 0, a0_max = 4095;
const int a1_min = 0, a1_max = 4095;
const int a2_min = 0,  a2_max = 4095;
int a0, a1, a2;


//==== GAMEPAD ====//
#define numOfButtons 0
#define numOfHatSwitches 0
#define enableX true
#define enableY true
#define enableZ true
#define enableRX false
#define enableRY false
#define enableRZ false
#define enableSlider1 false
#define enableSlider2 false

BleGamepad bleGamepad("Underwater coco 3", "CRC Mines", 100);

void setup() {
  Serial.begin(115200);

  // gamepad
  BleGamepadConfiguration bleGamepadConfig;
  bleGamepadConfig.setAutoReport(false);
  bleGamepadConfig.setControllerType(CONTROLLER_TYPE_GAMEPAD); // CONTROLLER_TYPE_JOYSTICK, CONTROLLER_TYPE_GAMEPAD (DEFAULT), CONTROLLER_TYPE_MULTI_AXIS
  bleGamepadConfig.setButtonCount(numOfButtons);
  bleGamepadConfig.setWhichAxes(enableX, enableY, enableZ, enableRX, enableRY, enableRZ, enableSlider1, enableSlider2);      // Can also be done per-axis individually. All are true by default
  bleGamepadConfig.setHatSwitchCount(numOfHatSwitches);                                                                      // 1 by default
  // Some non-Windows operating systems and web based gamepad testers don't like min axis set below 0, so 0 is set by default
  bleGamepadConfig.setAxesMin(0x00);
  bleGamepadConfig.setAxesMax(0xFF);
  bleGamepad.begin(&bleGamepadConfig);
}

void loop() {
   a0 = a1 = a2 = 0;
  // 32 samples for averaging
  for(int i = 0;i < NB_SAMPLE;i++) {
    a0 += analogRead(A0);
    a1 += analogRead(A1);
    a2 += analogRead(A2);
  }
  a0 /= NB_SAMPLE;
  a1 /= NB_SAMPLE;
  a2 /= NB_SAMPLE;

  if (bleGamepad.isConnected()) {
    bleGamepad.setX(convert(a0, a0_min, a0_max));
    bleGamepad.setY(convert(a1, a1_min, a1_max));
    bleGamepad.setZ(convert(a2, a2_min, a2_max));
    bleGamepad.sendReport();
  }

  Serial.print("a0:");
  Serial.print(convert(a0, a0_min, a0_max));
  Serial.print("\ta1:");
  Serial.print(convert(a1, a1_min, a1_max));
  Serial.print("\ta2:");
  Serial.println(convert(a2, a2_min, a2_max));
  
  delay(5);
}

float mapfloat(float x, float in_min, float in_max, float out_min, float out_max) {
  return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min;
}

int convert(float input, float min, float max) {
  float val = mapfloat(input, min, max, -1, 1) / range;
  val = val > deadzone ? val - deadzone : (val < -deadzone ? val + deadzone : 0);
  val /= (1 - deadzone);
  return int(mapfloat(constrain(val, -1, 1), -1, 1, 0, 255));
}