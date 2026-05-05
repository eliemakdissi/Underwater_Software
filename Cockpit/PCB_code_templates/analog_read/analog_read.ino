#define A0 1
#define A1 2
#define A2 3
#define A3 4

void setup() {
  // put your setup code here, to run once:
  Serial.begin(115200);
  
  pinMode(A0, INPUT);
  pinMode(A1, INPUT);
  pinMode(A2, INPUT);
  pinMode(A3, INPUT);

  analogReadResolution(12);
}

void loop() {
  // put your main code here, to run repeatedly:
  int a0 = 0;
  int a1 = 0;
  int a2 = 0;
  int a3 = 0;

  // 32 samples for averaging
  for(int i = 0;i < 32;i++) {
    a0 += analogRead(A0);
    a1 += analogRead(A1);
    a2 += analogRead(A2);
    a3 += analogRead(A3);
  }

  Serial.print(a0);
  Serial.print('\t');
  Serial.print(a1);
  Serial.print('\t');
  Serial.print(a2);
  Serial.print('\t');
  Serial.print(a3);
  Serial.println("");
}
