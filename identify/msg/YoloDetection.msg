# YoloDetection.msg

string target_name

float32 confidence

# 目标框中心点位置（像素坐标）
int32 cx
int32 cy

# 原图尺寸
int32 image_height
int32 image_width

# 检测框坐标（左上角和右下角）
int32 x_min
int32 y_min
int32 x_max
int32 y_max