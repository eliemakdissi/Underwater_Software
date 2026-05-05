#include <FastLED.h>
#define NUM_LEDS 6
#define LED_PIN 11
#define BRIGHTNESS 255 // 64/255


#define SW0 10
#define SW1 42
#define SW2 41
#define SW3 40
#define SW4 39
#define SW5 38

CRGBArray<NUM_LEDS> leds;

void setup() {
  // put your setup code here, to run once:
  Serial.begin(115200);
  
  pinMode(SW0, INPUT_PULLDOWN);
  pinMode(SW1, INPUT_PULLDOWN);
  pinMode(SW2, INPUT_PULLDOWN);
  pinMode(SW3, INPUT_PULLDOWN);
  pinMode(SW4, INPUT_PULLDOWN);
  pinMode(SW5, INPUT_PULLDOWN);

  FastLED.addLeds<NEOPIXEL,LED_PIN>(leds, NUM_LEDS);
}

void loop() {
  // put your main code here, to run repeatedly:
  int sw0 = digitalRead(SW0);
  int sw1 = digitalRead(SW1);
  int sw2 = digitalRead(SW2);
  int sw3 = digitalRead(SW3);
  int sw4 = digitalRead(SW4);
  int sw5 = digitalRead(SW5);

  int hue = int(millis() * 0.06) % 255;
  leds[0] = CHSV(hue, 255, BRIGHTNESS * sw0);
  leds[1] = CHSV(hue, 255, BRIGHTNESS * sw1);
  leds[2] = CHSV(hue, 255, BRIGHTNESS * sw2);
  leds[3] = CHSV(hue, 255, BRIGHTNESS * sw3);
  leds[4] = CHSV(hue, 255, BRIGHTNESS * sw4);
  leds[5] = CHSV(hue, 255, BRIGHTNESS * sw5);

  FastLED.delay(33);

  Serial.print(sw0);
  Serial.print('\t');
  Serial.print(sw1);
  Serial.print('\t');
  Serial.print(sw2);
  Serial.print('\t');
  Serial.print(sw3);
  Serial.print('\t');
  Serial.print(sw4);
  Serial.print('\t');
  Serial.println(sw5);
}
