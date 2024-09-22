mkdir -p ~/Documents/log ~/Documents/resources
docker run -it --rm --network host \
 -v $HOME/Documents/resources:/root/xmop/resources \
 -v $HOME/Documents/log:/root/xmop/log \
  prabinrath/xmop_data_gen:latest bash