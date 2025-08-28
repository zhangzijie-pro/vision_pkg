#ifndef PID_H
#define PID_H

class PIDController {
    private:
        float setpoint;       // 期望值
        float actualValue;    // 实际值

        float kp;             // 比例增益
        float ki;             // 积分增益
        float kd;             // 微分增益

        // 误差
        float error;
        float last_error;     // 上次误差
        float prev_error;

        float last_dt;

        float integral;       // 积分项
        // float integral_limit; // 积分饱和限制

        float output;     // 输出
        float last_output;

        // 是否为增量式
        bool INCR_SELECT;

    public:
        PIDController(float kp, float ki, float kd, bool INCR_SELECT);
        float Compute(float setpoint, float measured_value, float dt=1.0f/30.0f);
    };



#endif