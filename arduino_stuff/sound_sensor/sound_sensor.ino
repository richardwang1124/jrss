#define SOUND_PIN A0

int bdd = 9600;
int sound_pin=11;
unsigned int list1[200];
void setup() {
  Serial.begin(bdd);
  int start = millis();
  //pinMode(A0, INPUT);
  for(int i=0; i<200 ; i++){
     list1[i]=analogRead(A0);
     //delay(10);
     //Serial.println(analogRead(A0));
  }
  delay(1000);
  //Serial.println("OUT");
  Serial.print(millis()-start);
}

void loop() {
  Serial.println("Done");
  delay(1000);

}
