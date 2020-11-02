#!bin/bash

path_updater=/usr/src/pi-updater/
path_workspace=/home/pi/workspace/

# generate update_log.txt file at workspace folder
sudo touch ${path_workspace}.update_log
sudo chmod 666 ${path_workspace}.update_log

# chmod version file to update
sudo chmod 666 ${path_updater}version

# save log with time
logger()
{
	msg=$1
	sudo echo -e `date`: $msg >> ${path_workspace}.update_log 
}

# check wifi connection and return boolean
check_wifi()
{
	tmp=`iwconfig wlan0|grep ESSID:off`
	num="${#tmp}" 
	if [ $num = "0" ]; then
		wlan=true
	else
		wlan=false
	fi
	echo $wlan
}

# check local updater version compare with latest github version 
check_version()
{
	#ver_latest=`sudo curl https://raw.githubusercontent.com/wsy8029/pi-updater/master/version`
	while [ "$ver_latest" == "" ]; do
		$(logger "[VERSION] cannot get latest version, retry...")
#		sudo python3 ${path_updater}led/on_red.py
    		sudo python3 ${path_updater}led/on_orange.py
		ver_latest=$(wget https://raw.githubusercontent.com/wsy8029/pi-updater/master/version -q -O -)
		sudo python3 ${path_updater}led/off.py
	done
	#ver_latest=$(wget https://raw.githubusercontent.com/wsy8029/pi-updater/master/version -q -O -)
	$(logger "[VERSION] latest version is $ver_latest")
	ver_local=$(<${path_updater}version)	
	$(logger "[VERSION] current local version is $ver_local")
	if [ "$ver_latest" == "$ver_local" ]; then
		latest=true
	else
		latest=false
	fi
	echo $latest
}

$(logger "========================= Update Process Start =========================")

# update config and code when wlan is true
while [ true ]; do

	wlan=$(check_wifi)
	if [ "$wlan" == "true" ]; then
		sudo python3 ${path_updater}led/on_orange.py
		$(logger "[WIFI] wifi enable")
		latest=$(check_version)
		if [ $latest == true ]; then
			$(logger "[VERSION] local version is already up to date.")
#			sudo python3 ${path_updater}led/blink_rgb1.py
      			sudo python3 ${path_updater}led/on_blue.py
			break
		else
			$(logger "[VERSION] local version is older then latest version. Start update.")
			cd /usr/src/pi-updater/
			$(logger "[UPDATE] git pull...")
			sudo git fetch --all
			sudo git reset --hard origin/master
			sudo git pull origin master
			$(logger "[UPDATE] git pull completed")
			sudo /bin/bash ${path_updater}update_config.sh
			$(logger "[UPDATE] config updated")
			sudo /bin/bash ${path_updater}update_code.sh
			$(logger "[UPDATE] code updated")
			ver_updated=$(wget https://raw.githubusercontent.com/wsy8029/pi-updater/master/version -q -O -)
			$(logger "[UPDATE] Update complete (ver : $ver_updated )" )
			sudo echo $ver_updated > ${path_updater}version
			sudo python3 ${path_updater}led/on_blue.py
			break
		fi
# 	else
# 		$(logger "[WIFI] wifi disable, trying to connect wifi...")
# 		sudo /bin/cp -f ${path_updater}wpa_supplicant.conf /etc/wpa_supplicant/wpa_supplicant.conf
# 		sudo wpa_cli -i wlan0 reconfigure
# 		sudo ifconfig wlan0 down
# 		sudo ifconfig wlan0 up
# 		sudo python3 ${path_updater}led/on_orange.py
# 		sleep 5	
# 		sudo python3 ${path_updater}led/off.py
# 		sleep 1
	fi
done
$(logger "========================= Update Process End =========================")
echo "Update Complete"
