#include "pid.hpp"

PIDController::PIDController(float kp, float ki, float kd, bool INCR_SELECT)
:kp(kp), ki(ki), kd(kd), INCR_SELECT(INCR_SELECT)
{
    setpoint = 0.0;
    actualValue = 0.0;

    error = 0.0;
    last_error = 0.0;
    prev_error = 0.0;

    integral = 0.0;
    // integral_limit = 0.0;

    last_dt = 0.0;

    output = 0.0;     // 初始化 output
    last_output = 0.0;
}

/*
for image: target_cx - img_x == error

*/
float PIDController::Compute(float setpoint, float measured_value, float dt=1.0f/30.0f){
    error = setpoint - measured_value;
    setpoint = setpoint;
    if(INCR_SELECT){
        integral = dt==0? (error * dt):(error);
        
        output = (kp*(error - last_error)) + (ki * integral) + (kd * (error - 2.0f * last_error + prev_error));
        last_output = output;

        prev_error = last_error;
        last_error = error;

        actualValue += output-last_output;
        last_dt = dt;
        return output - last_output;

    }else{
        integral += dt==0? (error * dt):(error);

        output = (kp * error) + (ki * integral) + (kd*(error - last_error) / dt);
        actualValue += output;

        last_error = error;
        return output;
    }
}

