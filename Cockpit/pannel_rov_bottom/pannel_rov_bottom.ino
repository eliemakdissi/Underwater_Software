#include "Adafruit_TinyUSB.h"
#include <FastLED.h>

// Gamepad Report Descriptor Template
#define TUD_HID_REPORT_MY_GAMEPAD(...) \
  HID_USAGE_PAGE ( HID_USAGE_PAGE_DESKTOP     )                 ,\
  HID_USAGE      ( HID_USAGE_DESKTOP_GAMEPAD  )                 ,\
  HID_COLLECTION ( HID_COLLECTION_APPLICATION )                 ,\
    /* Report ID if any */\
    __VA_ARGS__ \
    /* 8 bit X, Y, Z, Rz, Rx, Ry (min -127, max 127 ) */ \
    HID_USAGE_PAGE     ( HID_USAGE_PAGE_DESKTOP                 ) ,\
    HID_USAGE          ( HID_USAGE_DESKTOP_X                    ) ,\
    HID_USAGE          ( HID_USAGE_DESKTOP_Y                    ) ,\
    HID_USAGE          ( HID_USAGE_DESKTOP_Z                    ) ,\
    HID_USAGE          ( HID_USAGE_DESKTOP_SLIDER               ) ,\
    HID_LOGICAL_MIN    ( 0x81                                   ) ,\
    HID_LOGICAL_MAX    ( 0x7f                                   ) ,\
    HID_REPORT_SIZE    ( 8                                      ) ,\
    HID_REPORT_COUNT   ( 4                                      ) ,\
    HID_INPUT          ( HID_DATA | HID_VARIABLE | HID_ABSOLUTE ) ,\
    /* 3 bit Button Map */ \
    HID_USAGE_PAGE     ( HID_USAGE_PAGE_BUTTON                  ) ,\
    HID_USAGE_MIN      ( 1                                      ) ,\
    HID_USAGE_MAX      ( 3                                      ) ,\
    HID_LOGICAL_MIN    ( 0                                      ) ,\
    HID_LOGICAL_MAX    ( 1                                      ) ,\
    HID_REPORT_SIZE    ( 1                                      ) ,\
    HID_REPORT_COUNT   ( 8                                      ) ,\
    HID_INPUT          ( HID_DATA | HID_VARIABLE | HID_ABSOLUTE ) ,\
  HID_COLLECTION_END \

// USB HID object
Adafruit_USBD_HID usb_hid;

uint8_t const desc_hid_report[] = {
    TUD_HID_REPORT_MY_GAMEPAD()
};

struct descriptor{
  int8_t  jx;      // Thumb joystick X
  int8_t  jy;      // Thumb joystick Y
  int8_t  yaw;     // Yaw command
  int8_t  wheel;   // Tilt wheel
  uint8_t buttons; // Buttons 
};

descriptor gp;

#define NUM_LEDS 6
#define LED_PIN 11
#define BRIGHTNESS 255 // 64/255


#define SW0 10
#define SW1 42
#define SW2 41
#define SW3 40
#define SW4 39
#define SW5 38

#define A0 1
#define A1 2
#define A2 3
#define A3 4
#define NB_SAMPLE 32

const int a0_min = 1534, a0_max = 2353;
const int a1_min = 1643, a1_max = 2324;
const int a2_min = 542,  a2_max = 3271;
const int	a3_min = 697,  a3_max = 3662;
const float deadzone = 0.15;
const float range = 0.95;

CRGBArray<NUM_LEDS> leds;

int mode = 0;

void setup() {
  TinyUSBDevice.setManufacturerDescriptor("Underwater");
  TinyUSBDevice.setProductDescriptor("2026_ROV_bottom");

  // Manual begin() is required on core without built-in support e.g. mbed rp2040
  if (!TinyUSBDevice.isInitialized()) {
    TinyUSBDevice.begin(0);
  }

  // put your setup code here, to run once:
  Serial.begin(115200);
  
  pinMode(SW0, INPUT_PULLDOWN);
  pinMode(SW1, INPUT_PULLDOWN);
  pinMode(SW2, INPUT_PULLDOWN);
  pinMode(SW3, INPUT_PULLDOWN);
  pinMode(SW4, INPUT_PULLDOWN);
  pinMode(SW5, INPUT_PULLDOWN);

  pinMode(A0, INPUT);
  pinMode(A1, INPUT);
  pinMode(A2, INPUT);
  pinMode(A3, INPUT);
  analogReadResolution(12);

  FastLED.addLeds<NEOPIXEL,LED_PIN>(leds, NUM_LEDS);

  // Setup HID
  usb_hid.setPollInterval(2);
  usb_hid.setReportDescriptor(desc_hid_report, sizeof(desc_hid_report));
  usb_hid.begin();

  // If already enumerated, additional class driverr begin() e.g msc, hid, midi won't take effect until re-enumeration
  if (TinyUSBDevice.mounted()) {
    TinyUSBDevice.detach();
    delay(10);
    TinyUSBDevice.attach();
  }
}


float mapfloat(float x, float in_min, float in_max, float out_min, float out_max) {
  return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min;
}


int convert(int input, int min, int max) {
  float val = mapfloat(input, min*NB_SAMPLE, max*NB_SAMPLE, -1, 1) / range;
  val = val > deadzone ? val - deadzone : (val < -deadzone ? val + deadzone : 0);
  val /= (1 - deadzone);
  return constrain(val * 127, -127, 127);
}

void loop() {
  if (!TinyUSBDevice.mounted()) {
    return;
  }
  if (!usb_hid.ready()) return;
  
  // put your main code here, to run repeatedly:
  int sw0 = digitalRead(SW0);
  int sw1 = digitalRead(SW1);
  int sw2 = digitalRead(SW2);
  int sw3 = digitalRead(SW3);
  int sw4 = digitalRead(SW4);
  int sw5 = digitalRead(SW5);

  int a0 = 0;
  int a1 = 0;
  int a2 = 0;
  int a3 = 0;

  // 32 samples for averaging
  for(int i = 0;i < NB_SAMPLE;i++) {
    a0 += analogRead(A0);
    a1 += analogRead(A1);
    a2 += analogRead(A2);
    a3 += analogRead(A3);
  }

  if(sw0) mode = 0;
  if(sw1) mode = 1;
  if(sw2) mode = 2;


  gp.yaw = convert(a1, a1_min, a1_max);
  gp.jy = convert(a3, a3_min, a3_max);
  gp.jx = convert(a2, a2_min, a2_max);
  gp.wheel = convert(a0, a0_min, a0_max);
  gp.buttons = sw0 | (sw1<<1) | (sw2<<2);
  usb_hid.sendReport(0, &gp, sizeof(gp));



  // feedback
  int hue = int(millis() * 0.06) % 255;
  leds[0] = CHSV(hue, 255, BRIGHTNESS * (mode == 0));
  leds[1] = CHSV(hue, 255, BRIGHTNESS * (mode == 1));
  leds[2] = CHSV(hue, 255, BRIGHTNESS * (mode == 2));
  leds[3] = CHSV(hue, 255, 0);
  leds[4] = CHSV(hue, 255, 0);
  leds[5] = CHSV(hue, 255, 0);

  FastLED.show();
  delay(5);
}
