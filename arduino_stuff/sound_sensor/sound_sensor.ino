#define SOUND_PIN A0

int sound_pin=11;

void setup() {
  Serial.begin(9600);
}

void loop() {
  int sensorState=analogRead(SOUND_PIN);
  Serial.println(sensorState);
}
