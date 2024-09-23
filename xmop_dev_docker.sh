xhost +
docker run --rm -it --gpus all --network=host --env DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v ./:/root/xmop prabinrath/xmop_dev:latest bash
