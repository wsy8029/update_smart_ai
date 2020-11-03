import shutil


# 업데이트할 폴더를 삭제 후 복사하는 방식으로 업데이
shutil.rmtree('/home/pi/workspace/ai-contents-gyro-car/src')
shutil.copytree('./ai-contents-gyro-car-master/src', '/home/pi/workspace/ai-contents-gyro-car/src')


# 자동 업데이터의 updater.sh 코드를 수정된 updater.sh 코드로 변경
shutil.copy('./updater.sh', '/usr/src/pi-updater/updater.sh')

print("업데이트가 완료되었습니다. Smart AI Kit을 재시작해주세요.")

# pymodi 1.0.1 설치를 위한 requirements - pexpect


# pip3 uninstall pymodi -y

# python3 setup.py install --user





