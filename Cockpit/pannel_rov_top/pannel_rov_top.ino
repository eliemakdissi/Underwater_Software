#include "Adafruit_TinyUSB.h"
#include <FastLED.h>
#include <RotaryEncoder.h>


//==== E-INK DISPLAY ====//
#include <GxEPD2_BW.h>
#include <Fonts/FreeMono9pt7b.h>
#include <Fonts/FreeMonoBold9pt7b.h>

GxEPD2_BW<GxEPD2_266_BN, GxEPD2_266_BN::HEIGHT> display( GxEPD2_266_BN(44, 43, 8, 39)); // CS, DC, RST, BUSY

#include "logo_minotaure_bitmap.h"
#include "logo_mines_bitmap.h"

char *button_names[] = {"trim x", "gain trans.", "trim y", "gain yaw", "trim z", "super gain", "trim yaw"};
bool prev_state_a0;
bool prev_state_a1;
bool prev_state_a2;
bool prev_state_a3;
int prev_val_a0;
int prev_val_a1;
int prev_val_a2;
int prev_val_a3;

int n_lines, padding;
float fspacing;

/*// Gamepad Report Descriptor Template
#define TUD_HID_REPORT_MY_GAMEPAD(...) \
  HID_USAGE_PAGE ( HID_USAGE_PAGE_DESKTOP     )                 ,\
  HID_USAGE      ( HID_USAGE_DESKTOP_GAMEPAD  )                 ,\
  HID_COLLECTION ( HID_COLLECTION_APPLICATION )                 ,\
    /* Report ID if any *\
    __VA_ARGS__ \
    /* 8 bit X, Y, Z, Rz, Rx, Ry (min -127, max 127 ) * \
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
    /* 3 bit Button Map * \
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

descriptor gp;*/


//==== LEDS ====//
#define NUM_LEDS 19
#define LED_PIN 11
#define BRIGHTNESS 255 //255
CRGB leds[NUM_LEDS];


//==== SWITCHS ====//
#define SW0 10
#define SW1 42
#define SW2 41
#define SW3 40
bool sw0_on = false;
bool sw1_on = false;
bool sw2_on = false;
bool sw3_on = false;
bool sw0_state = false;
bool sw1_state = false;
bool sw2_state = false;
bool sw3_state = false;

#define BUTTON 38
bool button_state;
long button_timer;
const int RAMP_UP_TIME = 500;
const int BOOST_TIME = 3000;
const int COOL_DOWN = 10000;


//==== POTENTIOMETERS ====//
#define A0 1
#define A1 2
#define A2 3
#define A3 4
#define NB_SAMPLE 32

const float deadzone = 0.05;
const float range = 0.95;
const int a0_min = 0, a0_max = 4095;
const int a1_min = 0, a1_max = 4095;
const int a2_min = 0,  a2_max = 4095;
const int	a3_min = 0,  a3_max = 4095;
int a0, a1, a2, a3;

//==== ROTARY ENCODER ====//
#define ENC1_A 12
#define ENC1_B 13
#define ENC2_A 5
#define ENC2_B 6
RotaryEncoder *enc1 = nullptr;
RotaryEncoder *enc2 = nullptr;
int gain1 = 0, gain2 = 0;
int prev_pos1 = 0, prev_pos2 = 0;

IRAM_ATTR void update_enc1() {
  enc1->tick();
}

IRAM_ATTR void update_enc2() {
  enc2->tick();
}




#define PERIOD 15;// target refresh rate : 66.7 Hz
unsigned long timer;

void setup() {
  /*TinyUSBDevice.setManufacturerDescriptor("Underwater");
  TinyUSBDevice.setProductDescriptor("2026_ROV_bottom");

  // Manual begin() is required on core without built-in support e.g. mbed rp2040
  if (!TinyUSBDevice.isInitialized()) {
    TinyUSBDevice.begin(0);
  }*/

  // put your setup code here, to run once:
  Serial.begin(115200);
  
  pinMode(SW0, INPUT_PULLDOWN);
  pinMode(SW1, INPUT_PULLDOWN);
  pinMode(SW2, INPUT_PULLDOWN);
  pinMode(SW3, INPUT_PULLDOWN);
  pinMode(BUTTON, INPUT_PULLDOWN);
  button_timer = -(RAMP_UP_TIME + BOOST_TIME + COOL_DOWN);
  
  pinMode(A0, INPUT);
  pinMode(A1, INPUT);
  pinMode(A2, INPUT);
  pinMode(A3, INPUT);
  analogReadResolution(12);

  // encoder 1
  enc1 = new RotaryEncoder(ENC1_A, ENC1_B, RotaryEncoder::LatchMode::FOUR0);
  pinMode(ENC1_A, INPUT_PULLDOWN);
  pinMode(ENC1_B, INPUT_PULLDOWN);
  attachInterrupt(digitalPinToInterrupt(ENC1_A), update_enc1, CHANGE);
  attachInterrupt(digitalPinToInterrupt(ENC1_B), update_enc1, CHANGE);
  // encoder 2
  enc2 = new RotaryEncoder(ENC2_A, ENC2_B, RotaryEncoder::LatchMode::FOUR0);
  pinMode(ENC2_A, INPUT_PULLDOWN);
  pinMode(ENC2_B, INPUT_PULLDOWN);
  attachInterrupt(digitalPinToInterrupt(ENC2_A), update_enc2, CHANGE);
  attachInterrupt(digitalPinToInterrupt(ENC2_B), update_enc2, CHANGE);

  // leds
  FastLED.addLeds<WS2812B,LED_PIN>(leds, NUM_LEDS);

  //e-ink display
  display.epd2.setBusyCallback(read_and_show_leds);
  display.init(0, true, 2, false);
  n_lines = sizeof(button_names) / sizeof(char*);
  padding = 10;
  fspacing = (display.height() - padding) / float(n_lines);

  /*// Setup HID
  usb_hid.setPollInterval(2);
  usb_hid.setReportDescriptor(desc_hid_report, sizeof(desc_hid_report));
  usb_hid.begin();

  // If already enumerated, additional class driverr begin() e.g msc, hid, midi won't take effect until re-enumeration
  if (TinyUSBDevice.mounted()) {
    TinyUSBDevice.detach();
    delay(10);
    TinyUSBDevice.attach();
  }*/


  timer = millis();


  // startup screen
  logo_minotaure();
  logo_mines();
  background();
  show_epd();
  display.hibernate();
}


float mapfloat(float x, float in_min, float in_max, float out_min, float out_max) {
  return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min;
}

int convert(int input, int min, int max) {
  float val = mapfloat(input, min*NB_SAMPLE, max*NB_SAMPLE, -1, 1) / range;
  val = val > deadzone ? val - deadzone : (val < -deadzone ? val + deadzone : 0);
  val /= (1 - deadzone);
  return constrain(-val * 127, -127, 127);
}

void loop() {
  read_and_show_leds(0);
  show_epd();
}

void wait(unsigned long delay) {
  unsigned long timer = millis() + delay;
  while(millis() < timer) {
    read_and_show_leds(0);
  }
}

void read_and_show_leds(const void *arg) {
  if(micros() < timer) return;
  timer += PERIOD;

  /*if (!TinyUSBDevice.mounted()) {
    return;
  }
  if (!usb_hid.ready()) return;*/
  
  bool sw0 = digitalRead(SW0);
  bool sw1 = digitalRead(SW1);
  bool sw2 = digitalRead(SW2);
  bool sw3 = digitalRead(SW3);
  if(sw0 != sw0_state && sw0) sw0_on = !sw0_on;
  if(sw1 != sw1_state && sw1) sw1_on = !sw1_on;
  if(sw2 != sw2_state && sw2) sw2_on = !sw2_on;
  if(sw3 != sw3_state && sw3) sw3_on = !sw3_on;
  sw0_state = sw0;
  sw1_state = sw1;
  sw2_state = sw2;
  sw3_state = sw3;

  bool button = digitalRead(BUTTON);
  if(button != button_state) {
    long time = millis() - button_timer;
    if(button && time > COOL_DOWN + RAMP_UP_TIME + BOOST_TIME) {
      button_timer = millis();
    }
    if(!button && time < RAMP_UP_TIME) {
      button_timer = millis() - (COOL_DOWN + RAMP_UP_TIME + BOOST_TIME);
    }
  }
  button_state = button;

  a0 = a1 = a2 = a3 = 0;
  // 32 samples for averaging
  for(int i = 0;i < NB_SAMPLE;i++) {
    a0 += analogRead(A0);
    a1 += analogRead(A1);
    a2 += analogRead(A2);
    a3 += analogRead(A3);
  }

  int enc_pos1 = enc1->getPosition();
  gain1 = constrain(gain1 + (enc_pos1 - prev_pos1), 0, 4);
  prev_pos1 = enc_pos1;

  int enc_pos2 = enc2->getPosition();
  gain2 = constrain(gain2 - (enc_pos2 - prev_pos2), 0, 4);
  prev_pos2 = enc_pos2;

  /*gp.yaw = convert(a1, a1_min, a1_max);
  gp.jy = convert(a3, a3_min, a3_max);
  gp.jx = convert(a2, a2_min, a2_max);
  gp.wheel = convert(a0, a0_min, a0_max);
  gp.buttons = sw0 | (sw1<<1) | (sw2<<2);
  usb_hid.sendReport(0, &gp, sizeof(gp));*/



  // led feedback
  int hue = int(millis() * 0.06) % 255;
  leds[0] = sw0_on ? CRGB::Green : CRGB::Black;
  leds[1] = sw1_on ? CRGB::Green : CRGB::Black;
  leds[2] = sw2_on ? CRGB::Green : CRGB::Black;
  leds[3] = sw3_on ? CRGB::Green : CRGB::Black;


  long button_time = millis() - button_timer;
  if(button_time < RAMP_UP_TIME) {
    int level = map(button_time, 0, RAMP_UP_TIME, 0, 4);
    for(int i = 0;i < 5;i++) {
      int hue = map(i, 0, 4, 0, 92);
      leds[4+i] = CHSV(hue, 255, (i <= level) * 128);
    }
  }else if(button_time-RAMP_UP_TIME < BOOST_TIME) {
    bool on = (button_time-RAMP_UP_TIME) % 250 < 125;
    for(int i = 0;i < 5;i++) {
      leds[4+i] = on ? CRGB::Green : CRGB::Black;
    }
  }else if(button_time-RAMP_UP_TIME-BOOST_TIME < COOL_DOWN) {
    bool on = abs(abs((button_time-RAMP_UP_TIME-BOOST_TIME) % 1000 - 200) - 100) > 50;
    int level = max(0l, map(button_time-RAMP_UP_TIME-BOOST_TIME, 0, COOL_DOWN, 4, -1));
    for(int i = 0;i < 5;i++) {
      int hue = map(i, 0, 4, 0, 92);
      leds[4+i] = CHSV(hue, 255, (i <= level && on) * 128);
    }
  }else{
    for(int i = 0;i < 5;i++) {
      leds[4+i] = CRGB::Black;
    }
  }

  for(int i = 0;i < 5;i++) {
    int hue = map(i, 0, 4, 0, 92);
    leds[13-i] = CHSV(hue, 255, (i <= gain1) * 128);
    leds[14+i] = CHSV(hue, 255, (i <= gain2) * 128);
  }
  
  FastLED.show();
  delayMicroseconds(500);// don't go lower than this, weird things happen to the leds
}

void show_epd() {
  uint16_t y = int(padding + fspacing / 2);
  uint16_t spacing = int(fspacing);

  if(abs(prev_val_a0 - a0) > 10*NB_SAMPLE || prev_state_a0 != sw3_on) {
    if(abs(prev_val_a0 - a0) > 10*NB_SAMPLE) prev_val_a0 = a0;
    if(prev_state_a0 != sw3_on) prev_state_a0 = sw3_on;

    float val = (convert(a0, a0_min, a0_max) + 127) / 255.0;
    drawBar(val, sw3_on, display.width()/2, y + 9, display.width() - 54, 7);
  }

  if(abs(prev_val_a1 - a1) > 10*NB_SAMPLE || prev_state_a1 != sw2_on) {
    if(abs(prev_val_a1 - a1) > 10*NB_SAMPLE) prev_val_a1 = a1;
    if(prev_state_a1 != sw2_on) prev_state_a1 = sw2_on;

    float val = (convert(a1, a1_min, a1_max) + 127) / 255.0;
    drawBar(val, sw2_on, display.width()/2, y + 9 + 2*spacing, display.width() - 54, 7);
  }

  if(abs(prev_val_a2 - a2) > 10*NB_SAMPLE || prev_state_a2 != sw1_on) {
    if(abs(prev_val_a2 - a2) > 10*NB_SAMPLE) prev_val_a2 = a2;
    if(prev_state_a2 != sw1_on) prev_state_a2 = sw1_on;

    float val = (convert(a2, a2_min, a2_max) + 127) / 255.0;
    drawBar(val, sw1_on, display.width()/2, y + 9 + 4*spacing, display.width() - 54, 7);
  }

  if(abs(prev_val_a3 - a3) > 10*NB_SAMPLE || prev_state_a3 != sw0_on) {
    if(abs(prev_val_a3 - a3) > 10*NB_SAMPLE) prev_val_a3 = a3;
    if(prev_state_a3 != sw0_on) prev_state_a3 = sw0_on;

    float val = (convert(a3, a3_min, a3_max) + 127) / 255.0;
    drawBar(val, sw0_on, display.width()/2, y + 9 + 6*spacing, display.width() - 54, 7);
  }
}

void drawBar(float val, bool active, int x, int y, uint w, uint h) {
  int fromx = x - w/2, fromy = y - h/2;

  int level_x = fromx + constrain(int(val * w), 0, w);

  display.setRotation(2);
  display.setPartialWindow(fromx, fromy, w, h);
  display.firstPage();
  do
  {
    display.fillScreen(GxEPD_WHITE);
    display.drawRect(fromx, fromy, w, h, GxEPD_BLACK);
    if(active) {
      display.fillRect(min(display.width()/2, level_x) + 1, fromy+1, max(abs(display.width()/2 - level_x + 1), 1), h-2, GxEPD_BLACK);
    }else{
      fillRectPattern(min(display.width()/2, level_x) + 1, fromy+1, max(abs(display.width()/2 - level_x + 1), 1), h-2);
    }
  }
  while (display.nextPage());
  display.hibernate();
}

void fillRectPattern(int x, int y, uint w, uint h) {
  for(int dx = 0;dx < w;dx++) {
    for(int dy = 0;dy < h;dy++) {
      bool on = (x+dx + y+dy) % 6 < 3;
      display.drawPixel(x+dx, y+dy, on ? GxEPD_BLACK : GxEPD_WHITE);
    }
  }
}

void background() {
  display.setRotation(2);
  display.setFullWindow();
  display.setFont(&FreeMonoBold9pt7b);
  display.setTextColor(GxEPD_BLACK);

  display.firstPage();
  do
  {
    display.fillScreen(GxEPD_WHITE);
    
    uint16_t y = int(padding + fspacing / 2);
    uint16_t spacing = int(fspacing);

    for(int i = 0;i < n_lines;i++) {
      int16_t d = int(abs(i-n_lines/2) * 4 + 1) * (int(i-n_lines/2 > 0) - int(i-n_lines/2 < 0));

      char* button = button_names[i];
      int16_t tbx, tby; uint16_t tbw, tbh;
      display.getTextBounds(button, 0, 0, &tbx, &tby, &tbw, &tbh);
  
      if((i & 0b1) == 0) {
        display.drawLine(0, y + d/2, abs(d), y, GxEPD_BLACK);
        display.drawLine(abs(d), y, 18, y, GxEPD_BLACK);
        display.drawLine(18, y-15, 18, y+15, GxEPD_BLACK);
        display.setCursor(22, y-4);
        display.print(button);
      }else{
        display.drawLine(display.width(), y + d/2, display.width()-abs(d), y, GxEPD_BLACK);
        display.drawLine(display.width()-abs(d), y, display.width()-18, y, GxEPD_BLACK);
        display.setCursor(display.width()-22-tbw, y+3);
        display.print(button);
        //pg.text(button_names[i], pg.width-22-w, y-5);
      }

      y += spacing;
    }
  }
  while (display.nextPage());
  display.hibernate();
  wait(1000);
}

const char HelloWorld[] = "Hello World!";

void logo_minotaure() {
  display.setRotation(2);
  display.setFullWindow();

  display.firstPage();
  do
  {
    display.fillScreen(GxEPD_WHITE);
    display.drawBitmap(0, 0, logo_minotaure_bitmap, 152, 296, GxEPD_BLACK);
  }
  while (display.nextPage());
  display.hibernate();
  wait(2000);
}

void logo_mines() {
  display.setRotation(2);
  display.setFullWindow();

  display.firstPage();
  do
  {
    display.fillScreen(GxEPD_WHITE);
    display.drawBitmap(0, 0, Logo_mines_bitmap, 152, 296, GxEPD_BLACK);
  }
  while (display.nextPage());
  display.hibernate();
  wait(2000);
}