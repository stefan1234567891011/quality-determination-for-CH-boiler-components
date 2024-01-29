import pyzed.sl as sl

class ZED2:
    zed = sl.Camera()

    def __init__(self, fps=30):
        #initialize resolition, etc
        self.init_params = sl.InitParameters()
        self.init_params.camera_resolution.HD1080
        self.init_params.camera_fps=fps

    def take_picture(self):
        # take picture
        err = self.zed.open(self.init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            print("could not initialize camera")
            return 1
        
        image = sl.Mat(mat_type=sl.MAT_TYPE.U8_C3)
        runtume_parameters = sl.RuntimeParameters()
        if self.zed.grab(runtume_parameters) == sl.ERROR_CODE.SUCCESS:
            self.zed.retrieve_image(image, sl.VIEW.LEFT)
            
        self.zed.close()
        return image