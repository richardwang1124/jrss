#include <Servo.h>

Servo myservo;

int pin = 11;
int ddd = 2;
void setup(){
 myservo.attach(pin);
}
void loop(){
  for(int i = 0;i<180;i++){
   myservo.write(i);
   delay(ddd);
  }
  for(int i = 180;i>0;i--){
   myservo.write(i);
   delay(ddd);
  }
}
