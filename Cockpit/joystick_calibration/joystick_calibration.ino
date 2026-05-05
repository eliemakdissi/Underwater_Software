#define A0 1
#define A1 2
#define A2 3
#define A3 4
#define NB_SAMPLE 512

int a0_min = 4096;
int a0_max = 0;
int a1_min = 4096;
int a1_max = 0;
int a2_min = 4096;
int a2_max = 0;
int a3_min = 4096;
int a3_max = 0;

void setup() {
  Serial.begin(115200);
  
  pinMode(A0, INPUT);
  pinMode(A1, INPUT);
  pinMode(A2, INPUT);
  pinMode(A3, INPUT);
  analogReadResolution(12);
}

void loop() {
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
  a0 /= NB_SAMPLE;
  a1 /= NB_SAMPLE;
  a2 /= NB_SAMPLE;
  a3 /= NB_SAMPLE;

  a0_min = min(a0_min, a0);
  a1_min = min(a1_min, a1);
  a2_min = min(a2_min, a2);
  a3_min = min(a3_min, a3);

  a0_max = max(a0_max, a0);
  a1_max = max(a1_max, a1);
  a2_max = max(a2_max, a2);
  a3_max = max(a3_max, a3);

  Serial.print("a0:");
  Serial.print(a0);
  Serial.print("\ta1:");
  Serial.print(a1);
  Serial.print("\ta2:");
  Serial.print(a2);
  Serial.print("\ta3:");
  Serial.println(a3);

  /*Serial.print("a0 = [");
  Serial.print(a0_min);
  Serial.print(":");
  Serial.print(a0_max);
  Serial.print("]\ta1 = [");
  Serial.print(a1_min);
  Serial.print(":");
  Serial.print(a1_max);
  Serial.print("]\ta2 = [");
  Serial.print(a2_min);
  Serial.print(":");
  Serial.print(a2_max);
  Serial.print("]\ta3 = [");
  Serial.print(a3_min);
  Serial.print(":");
  Serial.print(a3_max);
  Serial.println("]");*/
}
