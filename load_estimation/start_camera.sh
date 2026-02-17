#!/bin/bash
# Raspberry Pi Logitech C270 ROS2 camera startup script
# YUYV mode, manual exposure/gain, 320x240 @ 30 FPS

DEV=/dev/video2
EXPOSURE=20       # 50 x 100µs = 5 ms
GAIN=250
IMAGE_WIDTH=320
IMAGE_HEIGHT=240
FPS=30

echo "[INFO] Starting v4l2_camera node..."
ros2 run v4l2_camera v4l2_camera_node --ros-args \
  -p video_device:=$DEV \
  -p image_size:="[$IMAGE_WIDTH,$IMAGE_HEIGHT]" \
  -p frame_rate:=$FPS \
  -p pixel_format:="YUYV" \
  -p auto_exposure:=1 &

# Give the node a moment to start streaming
sleep 10

echo "[INFO] Applying manual exposure/gain..."
# Apply exposure/gain multiple times to ensure it sticks
for i in {1..3}; do
    v4l2-ctl -d $DEV --set-ctrl=auto_exposure=1
    v4l2-ctl -d $DEV --set-ctrl=exposure_time_absolute=$EXPOSURE
    v4l2-ctl -d $DEV --set-ctrl=gain=$GAIN
    sleep 0.2
done

echo "[INFO] Verifying camera controls..."
v4l2-ctl -d $DEV --get-ctrl=auto_exposure
v4l2-ctl -d $DEV --get-ctrl=exposure_time_absolute
v4l2-ctl -d $DEV --get-ctrl=gain

echo "[INFO] Camera ready. Stream available on /image_raw"
