# DpSTracker.msg

int32 id

string label

float32 confidence

# 目标框中心点位置（像素坐标）
float32 cx
float32 cy

# 检测框坐标（左上角和右下角）
float32 x_min
float32 y_min
float32 x_max
float32 y_max

float32 area_ratio