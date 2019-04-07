#include <Servo.h>

Servo myservo;

int pin = 11;
int down = 1;
int up = 7;
int test=0;

void setup() {
  Serial.begin(9600);
  myservo.attach(pin);

  while(Serial.read() == -1){}
}

void loop() {
  if(Serial.read() != -1){
    setup();
  }
  for(int i = 0;i<180;i++){
   myservo.write(i);
   delay(down);
  }
  for(int i = 180;i>0;i--){
   myservo.write(i);
   delay(up);
  }
  
}
