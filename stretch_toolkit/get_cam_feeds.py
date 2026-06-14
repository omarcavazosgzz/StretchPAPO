from dataclasses import dataclass
import pyrealsense2 as rs
import cv2
import numpy as np
import time

# D405 HELPERS
exposure_keywords = ['low', 'medium', 'auto']
exposure_range = [0, 500000]

def pixel_from_3d(xyz, camera_info):
    x_in, y_in, z_in = xyz
    camera_matrix = camera_info['camera_matrix']
    f_x = camera_matrix[0,0]
    c_x = camera_matrix[0,2]
    f_y = camera_matrix[1,1]
    c_y = camera_matrix[1,2]
    x_pix = ((f_x * x_in) / z_in) + c_x
    y_pix = ((f_y * y_in) / z_in) + c_y
    xy = np.array([x_pix, y_pix])
    return(xy)

def pixel_to_3d(xy_pix, z_in, camera_info):
    x_pix, y_pix = xy_pix
    camera_matrix = camera_info['camera_matrix']
    f_x = camera_matrix[0,0]
    c_x = camera_matrix[0,2]
    f_y = camera_matrix[1,1]
    c_y = camera_matrix[1,2]
    x_out = ((x_pix - c_x) * z_in) / f_x
    y_out = ((y_pix - c_y) * z_in) / f_y
    xyz_out = np.array([x_out, y_out, z_in])
    return(xyz_out)

def get_depth_scale(profile):
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    return depth_scale

def exposure_argument_is_valid(value):
    if value in exposure_keywords:
        return True
    is_string = isinstance(value, str)
    is_int = isinstance(value, int)
    int_value = exposure_range[0] - 10
    if is_string:
        if not value.isdigit():
            return False
        else:
            int_value = int(value)
    elif is_int:
        int_value = value
    if (int_value >= exposure_range[0]) or (int_value <= exposure_range[1]):
        return True
    return False

def check_exposure_value(value):
    if not exposure_argument_is_valid(value):
        raise ValueError(f'The provided exposure setting, {value}, is not a valide keyword, {exposure_keywords}, or is outside of the allowed numeric range, {exposure_range}.')    

def prepare_exposure_value(value):
    check_exposure_value(value)
    if value in exposure_keywords:
        return value
    is_int = isinstance(value, int)
    if is_int:
        return value
    is_string = isinstance(value, str)
    if is_string:
        return int(value)
    return None

def start_d405(exposure='auto'): 
    camera_info = [{'name': device.get_info(rs.camera_info.name),
                    'serial_number': device.get_info(rs.camera_info.serial_number)}
                   for device
                   in rs.context().devices]

    exposure = prepare_exposure_value(exposure)
    
    print('All cameras that were found:')
    print(camera_info)
    print()

    d405_info = None
    for info in camera_info:
        if info['name'].endswith('D405'):
            d405_info = info
    if d405_info is None:
        print('D405 camera not found')
        print('Exiting')
        exit()
    else:
        print('D405 found:')
        print(d405_info)
        print()

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(d405_info['serial_number'])
    
    # 1280 x 720, 5 fps
    # 848 x 480, 10 fps
    # 640 x 480, 30 fps

    #width, height, fps = 1280, 720, 5
    #width, height, fps = 848, 480, 10
    #width, height, fps = 640, 480, 30
    width, height, fps = 640, 480, 15
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

    profile = pipeline.start(config)
    
    if exposure == 'auto':
        # Use autoexposre
        stereo_sensor = pipeline.get_active_profile().get_device().query_sensors()[0]
        stereo_sensor.set_option(rs.option.enable_auto_exposure, True)
    else: 
        default_exposure = 33000
        if exposure == 'low':
            exposure_value = int(default_exposure/3.0)
        elif exposure == 'medium':
            exposure_value = 30000
        else:
            exposure_value = int(exposure)
            
        stereo_sensor = pipeline.get_active_profile().get_device().query_sensors()[0]
        stereo_sensor.set_option(rs.option.exposure, exposure_value)
    
    return pipeline, profile

# CAMERA FEED FUNCTIONS
last_head_frames = (None, None, 0)
last_wrist_frames = (None, None, 0)
def get_head_cam_frames():
    """ Get frames from the head camera (Intel RealSense D435i) """
    global last_head_frames
    get_head_cam_frames.pipeline = getattr(get_head_cam_frames, 'pipeline', None)
    
    if get_head_cam_frames.pipeline is None:
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 424, 240, rs.format.bgr8, 15)
        config.enable_stream(rs.stream.depth, 424, 240, rs.format.z16, 15)
        pipeline.start(config)
        get_head_cam_frames.pipeline = pipeline
    
    frames = get_head_cam_frames.pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()
    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())
    depth_image = cv2.rotate(depth_image, cv2.ROTATE_90_CLOCKWISE)
    color_image = cv2.rotate(color_image, cv2.ROTATE_90_CLOCKWISE)
    last_head_frames = (color_image, depth_image, time.time())
    return color_image, depth_image

def get_wrist_cam_frames():
    """ Get frames from the wrist camera (Intel RealSense D405) """
    global last_wrist_frames
    get_wrist_cam_frames.pipeline = getattr(get_wrist_cam_frames, 'pipeline', None)
    
    if get_wrist_cam_frames.pipeline is None:
        get_wrist_cam_frames.pipeline = start_d405(exposure='auto')[0]
    
    frames = get_wrist_cam_frames.pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()
    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())
    last_wrist_frames = (color_image, depth_image, time.time())
    return color_image, depth_image

def get_head_depth_frame():
    """Get depth frame from head camera, using cache if fresh."""
    global last_head_frames
    current_time = time.time()
    
    # Check if cached frame is fresh (< 1/60 second old)
    if last_head_frames[2] > 0 and (current_time - last_head_frames[2]) < 1/60:
        return last_head_frames[1]  # Return cached depth
    
    # Cache is stale, get fresh frames
    _, depth = get_head_cam_frames()
    return depth

def get_head_rgb_frame():
    """Get RGB frame from head camera, using cache if fresh."""
    global last_head_frames
    current_time = time.time()
    
    # Check if cached frame is fresh (< 1/60 second old)
    if last_head_frames[2] > 0 and (current_time - last_head_frames[2]) < 1/60:
        return last_head_frames[0]  # Return cached RGB
    
    # Cache is stale, get fresh frames
    rgb, _ = get_head_cam_frames()
    return rgb

def get_wrist_depth_frame():
    """Get depth frame from wrist camera, using cache if fresh."""
    global last_wrist_frames
    current_time = time.time()
    
    # Check if cached frame is fresh (< 1/60 second old)
    if last_wrist_frames[2] > 0 and (current_time - last_wrist_frames[2]) < 1/60:
        return last_wrist_frames[1]  # Return cached depth
    
    # Cache is stale, get fresh frames
    _, depth = get_wrist_cam_frames()
    return depth

def get_wrist_rgb_frame():
    """Get RGB frame from wrist camera, using cache if fresh."""
    global last_wrist_frames
    current_time = time.time()
    
    # Check if cached frame is fresh (< 1/60 second old)
    if last_wrist_frames[2] > 0 and (current_time - last_wrist_frames[2]) < 1/60:
        return last_wrist_frames[0]  # Return cached RGB
    
    # Cache is stale, get fresh frames
    rgb, _ = get_wrist_cam_frames()
    return rgb

def get_wide_cam_frames():
    get_wide_cam_frames.cap = getattr(get_wide_cam_frames, 'cap', None)
    
    if get_wide_cam_frames.cap is None:
        cap = cv2.VideoCapture(6)
        if not cap.isOpened():
            raise RuntimeError("Failed to open wide-angle camera at /dev/video6")
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        get_wide_cam_frames.cap = cap
    
    ret, frame = get_wide_cam_frames.cap.read()
    if ret:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return None

def stop_head_cam():
    """Stop and cleanup head camera pipeline"""
    if hasattr(get_head_cam_frames, 'pipeline') and get_head_cam_frames.pipeline:
        get_head_cam_frames.pipeline.stop()
        get_head_cam_frames.pipeline = None

def stop_wrist_cam():
    """Stop and cleanup wrist camera pipeline"""
    if hasattr(get_wrist_cam_frames, 'pipeline') and get_wrist_cam_frames.pipeline:
        get_wrist_cam_frames.pipeline.stop()
        get_wrist_cam_frames.pipeline = None

def stop_wide_cam():
    """Stop and cleanup wide-angle camera"""
    if hasattr(get_wide_cam_frames, 'cap') and get_wide_cam_frames.cap:
        get_wide_cam_frames.cap.release()
        get_wide_cam_frames.cap = None

def stop_all_cameras():
    """Stop and cleanup all cameras"""
    stop_head_cam()
    stop_wrist_cam()
    stop_wide_cam()



if __name__ == "__main__":
    try:
        while True:
            # Get camera feeds
            head_rgb, head_depth = get_head_cam_frames()
            wrist_rgb, wrist_depth = get_wrist_cam_frames()
            wide_rgb = get_wide_cam_frames()
            
            # Display RGB feeds
            cv2.imshow('Head Camera (D435)', head_rgb)
            cv2.imshow('Wrist Camera (D405)', wrist_rgb)
            cv2.imshow('Wide-Angle Camera (OV9782)', wide_rgb)
            
            # Exit on ESC key
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC key
                break
                
    finally:
        cv2.destroyAllWindows()
        stop_all_cameras()
