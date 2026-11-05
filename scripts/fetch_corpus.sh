#!/bin/bash
# Fetch the documentation corpus into data/corpus. Nothing here is redistributed by this repo.
# The ROS docs are cloned at the revisions the evaluation used. The three UR PDFs have to be downloaded by hand
# from universal-robots.com (free account / click-through), see README.
set -e
cd "$(dirname "$0")/.."
mkdir -p data/corpus/ur
clone() { [ -d "data/corpus/$1" ] || git clone -q --branch humble "$2" "data/corpus/$1"; git -C "data/corpus/$1" checkout -q "$3"; }
clone ros2_documentation https://github.com/ros2/ros2_documentation.git 109a8bb
clone moveit2_tutorials https://github.com/moveit/moveit2_tutorials.git b8f954c
clone Universal_Robots_ROS2_Driver https://github.com/UniversalRobots/Universal_Robots_ROS2_Driver.git 3d1fbe6
for f in UR3e_User_Manual_en_SW5.24.pdf UR3e_User_Manual_de_SW5.24.pdf ErrorCodes.pdf; do
  [ -f "data/corpus/ur/$f" ] || echo "missing data/corpus/ur/$f (download it from universal-robots.com, see README)"
done
