import pyzed.sl as sl

zed = sl.Camera()
init_params = sl.InitParameters()
init_params.camera_resolution = sl.RESOLUTION.HD720
init_params.depth_mode = sl.DEPTH_MODE.ULTRA

err = zed.open(init_params)
if err != sl.ERROR_CODE.SUCCESS:
    exit()

detection_parameters = sl.ObjectDetectionParameters()
detection_parameters.image_sync = True
detection_parameters.enable_tracking = True
detection_parameters.enable_mask_output = True

if detection_parameters.enable_tracking:
    zed.enable_positional_tracking()

print("Object Detection: Loading Module...")
err = zed.enable_object_detection(detection_parameters)
if err != sl.ERROR_CODE.SUCCES:
    print("Error {}, exit program")
    zed.close()
    exit()

detection_parameters_runtime = sl.ObjectDetectionRuntimeParameters()
detection_parameters_runtime.detection_confidence_threshold = 40

objects = sl.Objects()

while zed.grab() == sl.ERROR_CODE.SUCCES:
    err = zed.retrieve_objects(objects, detection_parameters_runtime)

    if objects.is_new:
        print(f"{0} Object(s) deetected" .format(len(objects.object_list)))

        if len(objects.object_list):
            first_object = objects.object_list[0]
            position = first_object.position
            print(f" 3D position : [{0}, {1}, {2}]" . format(position[0], position[1], position[2]))

            bounding_box = first_object.bounding_box
            print(" Bounding box 3D :")
            for it in bounding_box:
                print(" " + str(it), end='')

zed.disable_object_detection()
zed.close()
