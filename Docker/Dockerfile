
FROM ros:noetic-ros-base-focal

RUN useradd -g sudo relex --home-dir=/home/relex 
RUN echo "relex     ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers 
RUN mkdir /home/relex && chown relex /home/relex
WORKDIR /home/relex
user relex
 

RUN sudo apt-get update && sudo apt-get -y install cmake g++ libeigen3-dev libopencv-dev libboost-serialization-dev libopengl-dev libglew-dev libssh-dev wget
RUN sudo apt-get install unzip
RUN stat /home/relex
RUN wget https://github.com/stevenlovegrove/Pangolin/archive/master.zip &&\
  unzip master.zip && cd Pangolin-master && mkdir build && cd build && \
cmake .. && make -j2 && sudo make install && cd ~ && rm -rf Pangolin-master && rm master.zip

RUN wget https://github.com/noskill/ORB_SLAM3/archive/master.zip && unzip master.zip && rm master.zip 
RUN mkdir -p catkin_ws/src && cd catkin_ws && bash -c "source /opt/ros/noetic/setup.bash && catkin_make"
RUN mv ORB_SLAM3-master catkin_ws/src
RUN sudo apt-get -y install gnupg2 vim

RUN sudo apt-get -y install build-essential \
 debhelper-compat librosconsole-dev libsensor-msgs-dev libopencv-dev dh-python librostest-dev python3-numpy \
 libyaml-0-2 python3-genmsg python3-genpy python3-geometry-msgs python3-opencv python3-sensor-msgs python3-std-msgs python3-yaml
RUN wget https://github.com/noskill/ros-vision-opencv/archive/master.zip && \
   unzip master.zip && mv ros-vision-opencv-master catkin_ws/src 
RUN rm master.zip

RUN sudo apt-get -y install ros-noetic-tf ros-noetic-image-transport ros-noetic-image-geometry
RUN rosdep update && bash -c  "source /opt/ros/noetic/setup.bash \
&& cd catkin_ws && catkin_make"

# steps to build ORB_SLAM
# RUN bash -c  "source /opt/ros/noetic/setup.bash \
# && cd catkin_ws && source devel/setup.bash && \
#cd src/ORB_SLAM3-master && sh build.sh && sh build_ros.sh"
