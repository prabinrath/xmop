xhost +
docker run --rm -it --gpus all --network=host --env DISPLAY=$DISPLAY --device=/dev/dri:/dev/dri -v /tmp/.X11-unix:/tmp/.X11-unix -v ./:/root/xmop prabinrath/xmop_dev:latest bash