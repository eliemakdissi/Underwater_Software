// Set Tools > USB Stack > Adafruit TinyUSB


#include "Adafruit_TinyUSB.h"

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
    HID_USAGE          ( HID_USAGE_DESKTOP_RZ                   ) ,\
    HID_USAGE          ( HID_USAGE_DESKTOP_RX                   ) ,\
    HID_USAGE          ( HID_USAGE_DESKTOP_RY                   ) ,\
    HID_USAGE          ( HID_USAGE_DESKTOP_SLIDER               ) ,\
    HID_LOGICAL_MIN    ( 0x81                                   ) ,\
    HID_LOGICAL_MAX    ( 0x7f                                   ) ,\
    HID_REPORT_COUNT   ( 7                                      ) ,\
    HID_REPORT_SIZE    ( 8                                      ) ,\
    HID_INPUT          ( HID_DATA | HID_VARIABLE | HID_ABSOLUTE ) ,\
    /* 32 bit Button Map */ \
    HID_USAGE_PAGE     ( HID_USAGE_PAGE_BUTTON                  ) ,\
    HID_USAGE_MIN      ( 1                                      ) ,\
    HID_USAGE_MAX      ( 7                                      ) ,\
    HID_LOGICAL_MIN    ( 0                                      ) ,\
    HID_LOGICAL_MAX    ( 1                                      ) ,\
    HID_REPORT_COUNT   ( 7                                      ) ,\
    HID_REPORT_SIZE    ( 1                                      ) ,\
    HID_INPUT          ( HID_DATA | HID_VARIABLE | HID_ABSOLUTE ) ,\
  HID_COLLECTION_END \


uint8_t const desc_hid_report[] = {
    TUD_HID_REPORT_MY_GAMEPAD()
};


// USB HID object
Adafruit_USBD_HID usb_hid;

struct descriptor{
  int8_t  lx;        // X - left analog joystick
  int8_t  ly;        // Y - left analog joystick
  int8_t  lz;        // Z - left analog joystick
  int8_t  rz;        // X - left analog joystick
  int8_t  rx;        // Y - left analog joystick
  int8_t  ry;        // Z - left analog joystick
  int8_t  slider;    // Slider
  uint8_t buttons;   // Buttons 
};

descriptor gp;

void setup() {
  TinyUSBDevice.setManufacturerDescriptor("Mines");
  TinyUSBDevice.setProductDescriptor("2025_BDS_bottom");

  // Manual begin() is required on core without built-in support e.g. mbed rp2040
  if (!TinyUSBDevice.isInitialized()) {
    TinyUSBDevice.begin(0);
  }

  Serial.begin(115200);

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

  Serial.println("Adafruit TinyUSB HID Gamepad example");
}

void loop() {
  #ifdef TINYUSB_NEED_POLLING_TASK
  // Manual call tud_task since it isn't called by Core's background
  TinyUSBDevice.task();
  #endif

  // not enumerated()/mounted() yet: nothing to do
  if (!TinyUSBDevice.mounted()) {
    return;
  }

//  // Remote wakeup
//  if ( TinyUSBDevice.suspended() && btn )
//  {
//    // Wake up host if we are in suspend mode
//    // and REMOTE_WAKEUP feature is enabled by host
//    TinyUSBDevice.remoteWakeup();
//  }

  if (!usb_hid.ready()) return;

  // Reset buttons
  Serial.println("No pressing buttons");
  gp.lx = 0;
  gp.ly = 0;
  gp.lz = 0;
  gp.rz = 0;
  gp.rx = 0;
  gp.ry = 0;
  gp.slider = 0;
  gp.buttons = 0;
  usb_hid.sendReport(0, &gp, sizeof(gp));
  delay(2000);


  gp.slider = -64; 
  usb_hid.sendReport(0, &gp, sizeof(gp));
  delay(1000);

  gp.slider = 64;
  usb_hid.sendReport(0, &gp, sizeof(gp));
  delay(1000);

  gp.slider = 0;
  usb_hid.sendReport(0, &gp, sizeof(gp));
  delay(1000);


  // Joystick 1 UP
  Serial.println("Joystick 1 UP");
  gp.lx = 0;
  gp.ly = -127;
  usb_hid.sendReport(0, &gp, sizeof(gp));
  delay(1000);

  // Joystick 1 DOWN
  Serial.println("Joystick 1 DOWN");
  gp.lx = 0;
  gp.ly = 127;
  usb_hid.sendReport(0, &gp, sizeof(gp));
  delay(1000);

  // Joystick 1 RIGHT
  Serial.println("Joystick 1 RIGHT");
  gp.lx = 127;
  gp.ly = 0;
  usb_hid.sendReport(0, &gp, sizeof(gp));
  delay(1000);

  // Joystick 1 LEFT
  Serial.println("Joystick 1 LEFT");
  gp.lx = -127;
  gp.ly = 0;
  usb_hid.sendReport(0, &gp, sizeof(gp));
  delay(1000);

  // Joystick 1 CENTER
  Serial.println("Joystick 1 CENTER");
  gp.lx = 0;
  gp.ly = 0;
  usb_hid.sendReport(0, &gp, sizeof(gp));
  delay(1000);


  // Joystick 2 UP
  Serial.println("Joystick 2 UP");
  gp.lz = 0;
  gp.rz = 127;
  usb_hid.sendReport(0, &gp, sizeof(gp));
  delay(1000);

  // Joystick 2 DOWN
  Serial.println("Joystick 2 DOWN");
  gp.lz = 0;
  gp.rz = -127;
  usb_hid.sendReport(0, &gp, sizeof(gp));
  delay(1000);

  // Joystick 2 RIGHT
  Serial.println("Joystick 2 RIGHT");
  gp.lz = 127;
  gp.rz = 0;
  usb_hid.sendReport(0, &gp, sizeof(gp));
  delay(1000);

  // Joystick 2 LEFT
  Serial.println("Joystick 2 LEFT");
  gp.lz = -127;
  gp.rz = 0;
  usb_hid.sendReport(0, &gp, sizeof(gp));
  delay(2000);

  // Joystick 2 CENTER
  Serial.println("Joystick 2 CENTER");
  gp.lz = 0;
  gp.rz = 0;
  usb_hid.sendReport(0, &gp, sizeof(gp));
  delay(2000);


  // Analog Trigger 1 UP
  Serial.println("Analog Trigger 1 UP");
  gp.rx = 127;
  usb_hid.sendReport(0, &gp, sizeof(gp));
  delay(2000);

  // Analog Trigger 1 DOWN
  Serial.println("Analog Trigger 1 DOWN");
  gp.rx = -127;
  usb_hid.sendReport(0, &gp, sizeof(gp));
  delay(2000);

  // Analog Trigger 1 CENTER
  Serial.println("Analog Trigger 1 CENTER");
  gp.rx = 0;
  usb_hid.sendReport(0, &gp, sizeof(gp));
  delay(2000);


  // Analog Trigger 2 UP
  Serial.println("Analog Trigger 2 UP");
  gp.ry = 127;
  usb_hid.sendReport(0, &gp, sizeof(gp));
  delay(2000);

  // Analog Trigger 2 DOWN
  Serial.println("Analog Trigger 2 DOWN");
  gp.ry = -127;
  usb_hid.sendReport(0, &gp, sizeof(gp));
  delay(2000);

  // Analog Trigger 2 CENTER
  Serial.println("Analog Trigger 2 CENTER");
  gp.ry = 0;
  usb_hid.sendReport(0, &gp, sizeof(gp));
  delay(2000);


  // Test buttons (up to 32 buttons)
  for (int i = 0; i < 8; ++i) {
    Serial.print("Pressing button ");
    Serial.println(i);
    gp.buttons = (0b1 << i);
    usb_hid.sendReport(0, &gp, sizeof(gp));
    delay(1000);
  }

  // */
}

