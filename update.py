import shutil


# 업데이트할 폴더를 삭제 후 복사하는 방식으로 업데이트
# shutil.rmtree('/home/pi/workspace/ai-contents-gyro-car/src')
# shutil.copytree('./ai-contents-gyro-car-master/src', '/home/pi/workspace/ai-contents-gyro-car/src')
# 위 copytree 방식이 폴더 내 파일들이 0byte로 복사되는 문제가 있어, 개별 파일들을 복사하는 식으로 진행.
shutil.copy('/home/pi/workspace/update_smart_ai/ai-contents-gyro-car-master/src/main.ipynb', '/home/pi/workspace/ai-contents-gyro-car/src/main.ipynb')
shutil.copy('/home/pi/workspace/update_smart_ai/ai-contents-gyro-car-master/src/gesture_detection.py', '/home/pi/workspace/ai-contents-gyro-car/src/gesture_detection.py')
shutil.copy('/home/pi/workspace/update_smart_ai/ai-contents-gyro-car-master/src/gathering_data.py', '/home/pi/workspace/ai-contents-gyro-car/src/gathering_data.py')


# 자동 업데이터의 updater.sh 코드를 수정된 updater.sh 코드로 변경
shutil.copy('./updater.sh', '/usr/src/pi-updater/updater.sh')

# _ctrl_fan.py 파일을 수정된 코드로 변경
shutil.copy('./_ctrl_fan.py',  '/usr/src/rpi-daemon-py/_ctrl_fan.py')

print("업데이트가 완료되었습니다. Smart AI Kit을 재시작해주세요.")

# pymodi 1.0.1 설치를 위한 requirements - pexpect


# pip3 uninstall pymodi -y

# python3 setup.py install --user





