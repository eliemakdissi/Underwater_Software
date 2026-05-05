#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

#define SCREEN_WIDTH 128 // OLED display width, in pixels
#define SCREEN_HEIGHT 64 // OLED display height, in pixels
#define OLED_RESET    12 // Reset pin # (or -1 if sharing Arduino reset pin)
#define SCREEN_ADDRESS 0x3D ///< See datasheet for Address; 0x3D for 128x64, 0x3C for 128x32
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, OLED_RESET);

#include "logo_mines_de_paris_bitmap.h"
#include "logo_minotaure_bitmap.h"

void setup() {
  Serial.begin(115200);

  // Wait for display
  delay(500);

  // SSD1306_SWITCHCAPVCC = generate display voltage from 3.3V internally
  if(!display.begin(SSD1306_SWITCHCAPVCC, SCREEN_ADDRESS)) {
    Serial.println(F("SSD1306 allocation failed"));
    for(;;); // Don't proceed, loop forever
  }

  display.clearDisplay();
  display.drawBitmap(0, 0, logo_minotaure_bitmap, display.width(), display.height(), SSD1306_WHITE);
  display.display();
  delay(2000);
  display.clearDisplay();
  display.drawBitmap(0, 0, logo_mines_bitmap, display.width(), display.height(), SSD1306_WHITE);
  display.display();
  delay(2000);
}

void loop() {
  
}

void drawMenu() {
  
}
