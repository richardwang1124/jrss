#include <Stepper.h>
int t =0;
const int stepsPerRevolution = 200;  // change this to fit the number of steps per revolution
// for your motor


// initialize the stepper library on pins 8 through 11:
Stepper myStepper(stepsPerRevolution, 8, 9, 10, 11);

int stepCount = 0;  // number of steps the motor has taken
int motorSpeed=12;

void setup() s{
  Serial.begin(9600);
  myStepper.setSpeed(motorSpeed);

}

void loop() {
  if (motorSpeed > 0) {
    // step 1/100 of a revolution:
    myStepper.step(stepsPerRevolution / 25);
  Serial.println(motorSpeed);//print how fast the motor is going in the serial monitor
  }
}
