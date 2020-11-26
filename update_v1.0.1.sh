#!/bin/bash

echo -ne '###                         (5%)\r'
sleep 2

# Directory path
export PWD=${PWD}
export pexpect=$PWD/update_smart_ai/pexpect-master
export pymodi=$PWD/update_smart_ai/pymodi-pimodi

pushd $pexpect > /dev/null
python3 setup.py install --user --quiet > /dev/null
popd > /dev/null

echo -ne '#########                   (30%)\r'
sleep 2

pip3 uninstall pymodi -y > log.txt
echo -ne '##########                  (35%)\r'
sleep 2

pushd ${pymodi} > /dev/null
python3 setup.py install --user --quiet > /dev/null
popd > /dev/null
clear
echo -ne '###############             (50%)\r'
sleep 2


sudo rm /usr/src/rpi-daemon-py/_ctrl_fan.py
sudo cp $PWD/update_smart_ai/_ctrl_fan.py /usr/src/rpi-daemon-py/
echo -ne '#####################       (70%)\r'
sleep 2

sudo python3 /home/pi/workspace/update_smart_ai/update.py
echo '############################(100%)'
