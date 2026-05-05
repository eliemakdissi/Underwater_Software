#include <FastLED.h>
#include <RotaryEncoder.h>
#include <BleGamepad.h>
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

#define NUM_LEDS 6
#define LED_PIN 11
#define BRIGHTNESS 128 //255
CRGB leds[NUM_LEDS];


//==== SWITCHS ====//
#define SW0 10
#define SW1 42
#define SW2 41
#define SW3 40
#define SW4 39
#define SW5 38


//==== POTENTIOMETERS ====//
#define A0 1
#define A1 2
#define A2 3
#define A3 4
#define NB_SAMPLE 32

const float deadzone = 0.0;
const float range = 1.0;
const int a0_min = 0, a0_max = 4095;
const int a1_min = 0, a1_max = 4095;
const int a2_min = 0,  a2_max = 4095;
const int	a3_min = 0,  a3_max = 4095;
int a0, a1, a2, a3;


//==== ROTARY ENCODER ====//
#define ENC1_A 9
#define ENC1_B 8
#define ENC1_S 13
#define ENC2_A 7
#define ENC2_B 44
#define ENC2_S 43
RotaryEncoder *enc1 = nullptr;
RotaryEncoder *enc2 = nullptr;
int prev_enc1 = 0;
int prev_enc2 = 0;
bool prev_enc1_s = false;
bool prev_enc2_s = false;

IRAM_ATTR void update_enc1() { enc1->tick(); }
IRAM_ATTR void update_enc2() { enc2->tick(); }

//==== GAMEPAD ====//
#define numOfButtons 8
#define numOfHatSwitches 0
#define enableX true
#define enableY true
#define enableZ true
#define enableRX true
#define enableRY true
#define enableRZ true
#define enableSlider1 true
#define enableSlider2 true

BleGamepad bleGamepad("Underwater SLAM SONAR", "CRC Mines", 100);


//==== GUI ====//
#define SCREEN_WIDTH 128 // OLED display width, in pixels
#define SCREEN_HEIGHT 64 // OLED display height, in pixels
#define OLED_RESET    12 // Reset pin # (or -1 if sharing Arduino reset pin)
#define SCREEN_ADDRESS 0x3D ///< See datasheet for Address; 0x3D for 128x64, 0x3C for 128x32
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, OLED_RESET);

//#include "logo_mines_de_paris_bitmap.h"
//#include "logo_minotaure_bitmap.h"

#define NUM_PAGES 3
int active_page = 0;

const int tab_size = 16;
const int tab_spacing = 2;
const int menu_height = 6;

class Page {
public:
  String name;
  float inc;
  float val;
  float min, max;
  bool saved;

  Page() {}

  Page(String name, float inc, float initial_val, float min, float max) {
    this->name = name;
    this->inc = inc;
    this->min = min;
    this->max = max;
    this->val = initial_val;
    saved = false;
  }

  void increment(int count) {
    val = constrain(val + count * inc, min, max);
    saved = false;
  }

  void draw(bool pressed) {
    display.fillRect(0, menu_height + 1, display.width(), display.height() - menu_height - 1, SSD1306_BLACK);
    display.setTextColor(SSD1306_WHITE);
    display.setTextSize(1);

    // title
    display.setCursor(3, menu_height + 4);
    display.print(name);

    // value
    display.setTextSize(2);
    char buf[16];
    dtostrf(val, 0, 2, buf);  // float formatting
    int16_t x1, y1;
    uint16_t w, h;
    display.getTextBounds(buf, 0, 0, &x1, &y1, &w, &h);
    int x = (display.width() - w) / 2;
    int y = (display.height() - menu_height) / 2 + menu_height - 3;
    display.setCursor(x, y);
    display.print(buf);

    // "send" button
    const char* send = " send ";
    display.setTextSize(1);
    display.getTextBounds((char*)send, 0, 0, &x1, &y1, &w, &h);
    int bx = display.width() - w - 4;
    int by = display.height() - h;

    bool on = pressed || !(saved);
    display.setTextColor(!on ? SSD1306_WHITE : SSD1306_BLACK, on ? SSD1306_WHITE : SSD1306_BLACK);
    display.setCursor(bx, by);
    display.print(send);
  }

  int send() {
    saved = true;
    return convert(val, min, max);
  }
};

//==== PAGES ====//
Page pages[NUM_PAGES];

void setup() {
  Serial.begin(115200);

  pinMode(SW0, INPUT_PULLDOWN);
  pinMode(SW1, INPUT_PULLDOWN);
  pinMode(SW2, INPUT_PULLDOWN);
  pinMode(SW3, INPUT_PULLDOWN);
  pinMode(SW4, INPUT_PULLDOWN);
  pinMode(SW5, INPUT_PULLDOWN);
  
  // encoders
  enc1 = new RotaryEncoder(ENC1_A, ENC1_B, RotaryEncoder::LatchMode::FOUR0);
  pinMode(ENC1_A, INPUT_PULLUP);
  pinMode(ENC1_B, INPUT_PULLUP);
  pinMode(ENC1_S, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(ENC1_A), update_enc1, CHANGE);
  attachInterrupt(digitalPinToInterrupt(ENC1_B), update_enc1, CHANGE);

  enc2 = new RotaryEncoder(ENC2_A, ENC2_B, RotaryEncoder::LatchMode::FOUR0);
  pinMode(ENC2_A, INPUT_PULLUP);
  pinMode(ENC2_B, INPUT_PULLUP);
  pinMode(ENC2_S, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(ENC2_A), update_enc2, CHANGE);
  attachInterrupt(digitalPinToInterrupt(ENC2_B), update_enc2, CHANGE);

  // Setup controller with 16 buttons (plus start and select), accelerator, brake and steering
  BleGamepadConfiguration bleGamepadConfig;
  bleGamepadConfig.setAutoReport(false);
  bleGamepadConfig.setControllerType(CONTROLLER_TYPE_GAMEPAD); // CONTROLLER_TYPE_JOYSTICK, CONTROLLER_TYPE_GAMEPAD (DEFAULT), CONTROLLER_TYPE_MULTI_AXIS
  bleGamepadConfig.setVid(0xe502);
  bleGamepadConfig.setPid(0xabcd);
  bleGamepadConfig.setButtonCount(numOfButtons);
  bleGamepadConfig.setWhichAxes(enableX, enableY, enableZ, enableRX, enableRY, enableRZ, enableSlider1, enableSlider2);      // Can also be done per-axis individually. All are true by default
  bleGamepadConfig.setHatSwitchCount(numOfHatSwitches);                                                                      // 1 by default
  // Some non-Windows operating systems and web based gamepad testers don't like min axis set below 0, so 0 is set by default
  bleGamepadConfig.setAxesMin(0x0000);
  bleGamepadConfig.setAxesMax(0xFFFF);
  bleGamepad.begin(&bleGamepadConfig);

  // leds
  FastLED.addLeds<WS2812B,LED_PIN>(leds, NUM_LEDS);

  //screen
  if(!display.begin(SSD1306_SWITCHCAPVCC, SCREEN_ADDRESS)) {
    Serial.println(F("SSD1306 allocation failed"));
    for(;;); // Don't proceed, loop forever
  }
  display.clearDisplay();

  pages[0] = Page("sor nb neighbors", 1,   15, 5, 50);
  pages[1] = Page("sor std ratio", 0.01,   0.1, 0.01, 0.5);
  pages[2] = Page("hydro distance", 0.001, 0.565, 0.5, 0.65);

  drawMenu();
  pages[0].draw(false);
  display.display();
}

void loop() {
  // put your main code here, to run repeatedly:
  bool sw0 = digitalRead(SW0);
  bool sw1 = digitalRead(SW1);
  bool sw2 = digitalRead(SW2);
  bool sw3 = digitalRead(SW3);
  bool sw4 = digitalRead(SW4);
  bool sw5 = digitalRead(SW5);

   a0 = a1 = a2 = a3 = 0;
  // 32 samples for averaging
  for(int i = 0;i < NB_SAMPLE;i++) {
    a0 += analogRead(A0);
    a1 += analogRead(A1);
    a2 += analogRead(A2);
    a3 += analogRead(A3);
  }
  a0 /= NB_SAMPLE;
  a1 /= NB_SAMPLE;
  a2 /= NB_SAMPLE;
  a3 /= NB_SAMPLE;

  bool enc1_s = !digitalRead(ENC1_S);
  bool enc2_s = !digitalRead(ENC2_S);

  int enc1_pos = enc1->getPosition();
  int enc2_pos = enc2->getPosition();
  if(enc2_pos != prev_enc2) {
    active_page = (active_page + NUM_PAGES + (enc2_pos - prev_enc2)) % NUM_PAGES;
    drawMenu();
    pages[active_page].draw(enc1_s);
  }
  if(enc1_pos != prev_enc1) {
    pages[active_page].increment(enc1_pos - prev_enc1);
    pages[active_page].draw(enc1_s);
  }
  if(enc1_s != prev_enc1_s) {
    pages[active_page].draw(enc1_s);
  }
  prev_enc1 = enc1_pos;
  prev_enc2 = enc2_pos;
  prev_enc1_s = enc1_s;
  prev_enc2_s = enc2_s;
  display.display();

  if (bleGamepad.isConnected()) {
    bleGamepad.resetButtons();
    if(sw0) bleGamepad.press(BUTTON_1);
    if(sw1) bleGamepad.press(BUTTON_2);
    if(sw2) bleGamepad.press(BUTTON_3);
    if(sw3) bleGamepad.press(BUTTON_4);
    if(sw4) bleGamepad.press(BUTTON_5);
    if(sw5) bleGamepad.press(BUTTON_6);
    bleGamepad.setX(convert(a0, a0_min, a0_max));
    bleGamepad.setY(convert(a1, a1_min, a1_max));
    bleGamepad.setZ(convert(a2, a2_min, a2_max));
    bleGamepad.setRX(convert(a3, a3_min, a3_max));
    bleGamepad.setRY(pages[0].send());
    bleGamepad.setRZ(pages[1].send());
    bleGamepad.setSlider1(pages[2].send());
    bleGamepad.setSlider2(0);
    bleGamepad.sendReport();
  }



  /*Serial.print("a0:");
  Serial.print(convert(a0, a0_min, a0_max));
  Serial.print("\ta1:");
  Serial.print(convert(a1, a1_min, a1_max));
  Serial.print("\ta2:");
  Serial.print(convert(a2, a2_min, a2_max));
  Serial.print("\ta3:");
  Serial.print(convert(a3, a3_min, a3_max));
  Serial.print("\tsw1:");
  Serial.print(enc1_s);
  Serial.print("\tsw2:");
  Serial.print(enc2_s);
  Serial.print("\te1:");
  Serial.print(enc1->getPosition());
  Serial.print("\te2:");
  Serial.println(enc2->getPosition());*/  

  int hue = int(millis() * 0.06) % 255;
  leds[0] = CHSV(hue, 255, BRIGHTNESS * sw0);
  leds[1] = CHSV(hue, 255, BRIGHTNESS * sw1);
  leds[2] = CHSV(hue, 255, BRIGHTNESS * sw2);
  leds[3] = CHSV(hue, 255, BRIGHTNESS * sw3);
  leds[4] = CHSV(hue, 255, BRIGHTNESS * sw4);
  leds[5] = CHSV(hue, 255, BRIGHTNESS * sw5);
  FastLED.show();
  
  delay(5);
}

float mapfloat(float x, float in_min, float in_max, float out_min, float out_max) {
  return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min;
}

int convert(float input, float min, float max) {
  float val = mapfloat(input, min, max, -1, 1) / range;
  val = val > deadzone ? val - deadzone : (val < -deadzone ? val + deadzone : 0);
  val /= (1 - deadzone);
  return int(mapfloat(constrain(val, -1, 1), -1, 1, 0, 65535));
}

void drawMenu() {
  display.fillRect(0, 0, SCREEN_WIDTH, menu_height + 1, SSD1306_BLACK);
  display.drawLine(0, menu_height, SCREEN_WIDTH, menu_height, SSD1306_WHITE);

  int x = 2;
  for (int i = 0; i < NUM_PAGES; i++) {
    bool active = (i == active_page);

    if(active) {
      display.drawRoundRect(x, 0, tab_size+1, menu_height, 3, SSD1306_WHITE);
    }else{
      display.fillRoundRect(x, 0, tab_size+1, menu_height, 3, SSD1306_WHITE);
    }
    display.drawFastVLine(x, menu_height/2, menu_height/2, SSD1306_WHITE);
    display.drawFastVLine(x+tab_size, menu_height/2, menu_height/2, SSD1306_WHITE);

    display.fillRect(x+1, menu_height/2, tab_size-1, menu_height/2+1, active ? SSD1306_BLACK : SSD1306_WHITE);

    x += tab_size + tab_spacing;
  }
}
