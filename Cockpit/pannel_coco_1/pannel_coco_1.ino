#include <FastLED.h>
#include <RotaryEncoder.h>
#include <BleGamepad.h>

#define NUM_LEDS 4
#define LED_PIN 11
#define BRIGHTNESS 128 //255
CRGB leds[NUM_LEDS];


//==== SWITCHS ====//
#define SW0 41
#define SW1 40
#define SW2 39
#define SW3 38
#define S1A 9
#define S1B 8
#define S2A 12
#define S2B 13


//==== ROTARY ENCODER ====//
#define ENC1_A 7
#define ENC1_B 4
#define ENC2_A 6
#define ENC2_B 5
#define ENC3_A 42
#define ENC3_B 10
#define ENC4_A 3
#define ENC4_B 2
RotaryEncoder *enc1 = nullptr;
RotaryEncoder *enc2 = nullptr;
RotaryEncoder *enc3 = nullptr;
RotaryEncoder *enc4 = nullptr;

IRAM_ATTR void update_enc1() { enc1->tick(); }
IRAM_ATTR void update_enc2() { enc2->tick(); }
IRAM_ATTR void update_enc3() { enc3->tick(); }
IRAM_ATTR void update_enc4() { enc4->tick(); }


//==== GAMEPAD ====//
#define numOfButtons 8
#define numOfHatSwitches 0
#define enableX true
#define enableY true
#define enableZ true
#define enableRX true
#define enableRY false
#define enableRZ false
#define enableSlider1 false
#define enableSlider2 false

BleGamepad bleGamepad("Underwater coco 1", "CRC Mines", 100);

void setup() {
  Serial.begin(115200);

  pinMode(SW0, INPUT_PULLDOWN);
  pinMode(SW1, INPUT_PULLDOWN);
  pinMode(SW2, INPUT_PULLDOWN);
  pinMode(SW3, INPUT_PULLDOWN);
  pinMode(S1A, INPUT_PULLUP);
  pinMode(S1B, INPUT_PULLUP);
  pinMode(S2A, INPUT_PULLUP);
  pinMode(S2B, INPUT_PULLUP);
  
  // encoders
  enc1 = new RotaryEncoder(ENC1_A, ENC1_B, RotaryEncoder::LatchMode::FOUR0);
  pinMode(ENC1_A, INPUT_PULLUP);
  pinMode(ENC1_B, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(ENC1_A), update_enc1, CHANGE);
  attachInterrupt(digitalPinToInterrupt(ENC1_B), update_enc1, CHANGE);

  enc2 = new RotaryEncoder(ENC2_A, ENC2_B, RotaryEncoder::LatchMode::FOUR0);
  pinMode(ENC2_A, INPUT_PULLUP);
  pinMode(ENC2_B, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(ENC2_A), update_enc2, CHANGE);
  attachInterrupt(digitalPinToInterrupt(ENC2_B), update_enc2, CHANGE);

  enc3 = new RotaryEncoder(ENC3_A, ENC3_B, RotaryEncoder::LatchMode::FOUR0);
  pinMode(ENC3_A, INPUT_PULLUP);
  pinMode(ENC3_B, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(ENC3_A), update_enc3, CHANGE);
  attachInterrupt(digitalPinToInterrupt(ENC3_B), update_enc3, CHANGE);

  enc4 = new RotaryEncoder(ENC4_A, ENC4_B, RotaryEncoder::LatchMode::FOUR0);
  pinMode(ENC4_A, INPUT_PULLUP);
  pinMode(ENC4_B, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(ENC4_A), update_enc4, CHANGE);
  attachInterrupt(digitalPinToInterrupt(ENC4_B), update_enc4, CHANGE);


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

  // leds
  FastLED.addLeds<WS2812B,LED_PIN>(leds, NUM_LEDS);
}

void loop() {
  // put your main code here, to run repeatedly:
  bool sw0 = digitalRead(SW0);
  bool sw1 = digitalRead(SW1);
  bool sw2 = digitalRead(SW2);
  bool sw3 = digitalRead(SW3);
  bool s1A = !digitalRead(S1A);
  bool s1B = !digitalRead(S1B);
  bool s2A = !digitalRead(S2A);
  bool s2B = !digitalRead(S2B);

  if (bleGamepad.isConnected()) {
    bleGamepad.resetButtons();
    if(sw0) bleGamepad.press(BUTTON_1);
    if(sw1) bleGamepad.press(BUTTON_2);
    if(sw2) bleGamepad.press(BUTTON_3);
    if(sw3) bleGamepad.press(BUTTON_4);
    if(s1A) bleGamepad.press(BUTTON_5);
    if(s1B) bleGamepad.press(BUTTON_6);
    if(s2A) bleGamepad.press(BUTTON_7);
    if(s2B) bleGamepad.press(BUTTON_8);
    bleGamepad.setX(convert(enc1->getPosition(), 2));
    bleGamepad.setY(convert(enc2->getPosition(), 2));
    bleGamepad.setZ(convert(enc3->getPosition(), 2));
    bleGamepad.setRX(convert(enc4->getPosition(), 2));
    bleGamepad.sendReport();
  }

  int hue = int(millis() * 0.06) % 255;
  leds[0] = CHSV(hue, 255, BRIGHTNESS * sw0);
  leds[1] = CHSV(hue, 255, BRIGHTNESS * sw1);
  leds[2] = CHSV(hue, 255, BRIGHTNESS * sw2);
  leds[3] = CHSV(hue, 255, BRIGHTNESS * sw3);
  FastLED.show();

  delay(10);
}

uint convert(int x, int mode) {
  switch(mode) {
    case 1 :
    return constrain(x, 0, 255);
    break;

    case 2 :
    return constrain(x + 128, 0, 255);
    break;

    default :
    return (x % 256 + 256) % 256;
  }  
}