from hobot_vio import libsrcampy
import time

def test_camera_bind_display():
    #camera start
    cam = libsrcampy.Camera()
    ret = cam.open_cam(0, 1, 30, 1280, 720)
    print("Camera open_cam return:%d" % ret)

    #display start
    disp = libsrcampy.Display()
    ret = disp.display(0, 1920, 1080, 0, 1, chn_width = 1280, chn_height = 720)
    print ("Display display 0 return:%d" % ret)
    ret = disp.display(2, chn_width = 1280, chn_height = 720)
    print ("Display display 2 return:%d" % ret)
    disp.set_graph_rect(100, 100, 1920, 200, chn = 2, flush = 1,  color = 0xffff00ff)
    string = "horizon"
    disp.set_graph_word(300, 300, string.encode('gb2312'), 2, 0, 0xff00ffff)
    ret = libsrcampy.bind(cam, disp)
    print("libsrcampy bind return:%d" % ret)
    
    time.sleep(10)

    ret = libsrcampy.unbind(cam, disp)
    print("libsrcampy unbind return:%d" % ret)
    disp.close()
    cam.close_cam()
    print("test_camera_bind_display done!!!")

# open camera -> sleep(10) -> bind
test_camera_bind_display()