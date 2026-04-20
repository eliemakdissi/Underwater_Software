#include <GxEPD2_BW.h>
#include <Fonts/FreeMono9pt7b.h>
#include <Fonts/FreeMonoBold9pt7b.h>

GxEPD2_BW<GxEPD2_266_BN, GxEPD2_266_BN::HEIGHT> display( GxEPD2_266_BN(44, 43, 8, 39)); // CS, DC, RST, BUSY

#include "logo_minotaure_bitmap.h"
#include "logo_mines_bitmap.h"

#define PERIOD 15;// target refresh rate : 66.7 Hz
unsigned long timer;

void setup() {
  Serial.begin(115200);
  display.epd2.setBusyCallback(busyCallBack);
  display.init(0, true, 2, false);
  logo_minotaure();
  logo_mines();
  background();
  display.hibernate();
}

void loop() {
  
}

char *button_names[] = {"trim x", "gain trans.", "trim y", "gain yaw", "trim z", "super gain", "trim yaw"};

void background() {
  display.setRotation(2);
  display.setFullWindow();
  display.setFont(&FreeMonoBold9pt7b);
  display.setTextColor(GxEPD_BLACK);

  int n = sizeof(button_names) / sizeof(char*);
  int padding = 10;
  float fspacing = (display.height() - padding) / float(n);

  display.firstPage();
  do
  {
    display.fillScreen(GxEPD_WHITE);
    
    float fy = padding + fspacing / 2;
    for(int i = 0;i < n;i++) {
      int16_t d = int(abs(i-n/2) * 4 + 1) * (int(i-n/2 > 0) - int(i-n/2 < 0));
      uint16_t y = int(fy);
      uint16_t spacing = int(fspacing);

      char* button = button_names[i];
      int16_t tbx, tby; uint16_t tbw, tbh;
      display.getTextBounds(button, 0, 0, &tbx, &tby, &tbw, &tbh);
  
      if((i & 0b1) == 0) {
        display.drawLine(0, y + d/2, abs(d), y, GxEPD_BLACK);
        display.drawLine(abs(d), y, 18, y, GxEPD_BLACK);
        display.drawLine(18, y-15, 18, y+15, GxEPD_BLACK);
        display.setCursor(22, y-4);
        display.print(button);
        display.drawLine(22, y+10, display.width()-22, y+10, GxEPD_BLACK);
        display.drawLine(display.width()/2, y+7, display.width()/2, y+13, GxEPD_BLACK);
      }else{
        display.drawLine(display.width(), y + d/2, display.width()-abs(d), y, GxEPD_BLACK);
        display.drawLine(display.width()-abs(d), y, display.width()-18, y, GxEPD_BLACK);
        display.setCursor(display.width()-22-tbw, y+3);
        display.print(button);
        //pg.text(button_names[i], pg.width-22-w, y-5);
      }

      fy += spacing;
    }
  }
  while (display.nextPage());
  delay(1000);
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
  delay(2000);
}

void busyCallBack(const void *arg) {
  Serial.print("busy ");
  Serial.println(millis());
  delay(10);
  //(void) arg;
}

void logo_mines() {
  display.setRotation(2);
  display.setFullWindow();

  long timer = millis();
  display.firstPage();
  do
  {
    display.fillScreen(GxEPD_WHITE);
    display.drawBitmap(0, 0, Logo_mines_bitmap, 152, 296, GxEPD_BLACK);
  }
  while (display.nextPage());
  Serial.printf("diplay instruction time : %ul", millis() - timer);
  delay(2000);
}

void helloWorld() {
  display.setRotation(2);
  display.setFont(&FreeMonoBold9pt7b);
  display.setTextColor(GxEPD_BLACK);
  int16_t tbx, tby; uint16_t tbw, tbh;
  display.getTextBounds(HelloWorld, 0, 0, &tbx, &tby, &tbw, &tbh);
  // center the bounding box by transposition of the origin:
  uint16_t x = ((display.width() - tbw) / 2) - tbx;
  uint16_t y = ((display.height() - tbh) / 2) - tby;
  display.setFullWindow();
  display.firstPage();
  do
  {
    display.fillScreen(GxEPD_WHITE);
    display.setCursor(x, y);
    display.print(HelloWorld);
  }
  while (display.nextPage());
}