# Thymio Robot Documentation: A total beginners guide to the galaxy

This manual explains how to use the code on this git when working with our
equipment in the Robotlab of the VU in Amsterdam. We are currently working on a
more general manual and explanation of the equipment you need to start your own
experiments in your own lab. Many students use this code and develop it further.
We try to have the most recent version online but it might happen that some
explanations are double or not consistent. For questions you are welcome to
contact [j.v.heinerman AT vu.nl](j.v.heinerman@vu.nl)
		
## Setting Up

Make sure you have a UNIX environment terminal to execute the commands needed
to communicate with the Raspberry PI, e.g. Mac OSX or Linux and then connect
to the WiFi. 

* Channel:	ThymioNet
* Password:	`172luckytulip75B`

Download the appropriate packages to execute and test the robots use

    git clone https://github.com/ci-group/NEATThymio.git

## Logging In

Check the IP address on the back of the robot you want to use, e.g. `192.168.1.72`.
Connect to the robot with SSH

    ssh pi@<IP_addres>

You will be prompted for a password to connect ot Thymio

* Password:	`raspberry`

*NOTE:* To avoid having to put in your password everytime share your public SSH key
 with the PI

	ssh-keygen -t rsa
	ssh-copy-id pi@<IP_addres>

[1] See: [Passwordless SSH access](https://www.raspberrypi.org/documentation/remote-access/ssh/passwordless.md)


## Executing Scripts

### External execution

Pick your local booting script, e.g. `start_one.sh` and execute the script, e.g.
`foraging.py`.

	./start_one.sh foraging.py 192.168.1.72 1

### Local multiple execution

Pick your local booting script, e.g. `start_all.sh` and open `bots.txt`
 
    cd NEATThymio
    cd src
    vim bots.txt

Type in all IPs of PIs you want to start and make sure there is an empty line in
the bottom, e.g:
	
	192.168.1.52
	192.168.1.72
	192.168.1.62
	
Execute the script you want, e.g. `foraging.py`

	./start_all.sh foraging.py

## Synchronize 

Use `rsync` to post on the PI and pick a robot, e.g. `192.168.1.52`
Pick a file on your computer, e.g. `test.txt` and `pwd` the path.
`Rsync` it with the PI

	rsync <path_to>/test.txt pi@192.168.1.72:~/

Use `rsync` to pull from the PI

    rsync pi@<IP_address>:~/test.txt <path_to_copy>

## Clean Install
	
Get an SD-card, preferably 8GB or more and format it to FAT32.
Install [Raspbian image](https://www.raspberrypi.org/documentation/installation/installing-images/)
Find the correct IP address of the PI

    nmap -sn 192.168.1.*

Install the needed packages on PI

     sudo apt-get update
     sudo apt-get install rsync python-opencv python-numpy python-dbus python-gobject python-picamera dbus dbus-x11
     sudo apt-get install libboost-dev libqt4-dev qt4-dev-tools libqwt5-qt4-dev libudev-dev cmake g++ git make

*NOTE:* If the last one does not work try the following commands
followed by the above install again

     sudo apt-get clean
     sudo apt-get -f install

Download the [Aseba images](https://www.thymio.org/local--files/en:linuxinstall/aseba_1.4.0_armhf.deb)
and `rsync` the .deb file to the PI. Run the following command:

    sudo dpkg -i aseba_1.4.0_armhf.deb

## Setup PI Wifi
	
Stick a ETHERNET cable in your PI and find the correct IP address
of the PI

    nmap -sn 192.168.1.*

SSH with your PI and:

    vim /etc/network/interfaces

Replace file content with the following

	auto wlan0

	auto lo

	iface lo inet loopback
	iface eth0 inet dhcp

	allow-hotplug wlan0
	iface wlan0 inet static
	address 192.168.1.XX
	netmask 255.255.255.0
	gateway 192.168.1.1
	dns-nameservers 8.8.8.8
	wpa-conf /etc/wpa_supplicant/wpa_supplicant.conf
	iface default inet dhcp

*NOTE:* Replace the XX with your IP address

Also alter this file:

    vim /etc/wpa_supplicant/wpa_supplicant.conf

and replace file content with the following:

	ctrl_interface=DIR=/var/run/wpa_supplicant GROUP=netdev
	update_config=1

	network={
			ssid="ThymioNet"
			psk="172luckytulip75B"
			proto=RSN
			key_mgmt=WPA-PSK
			pairwise=CCMP
			auth_alg=OPEN
	}
			
## Using the logs
	
Go the log directory of your NEATThymio map and change `bots.txt` 
to include the IP(s) of the PI you want the logs off.

Run `get_logs.sh` locally

    ./get_logs.sh bots.txt

You will get an error that a map does not exist. Then create that
map locally and repeat previous step again:

    ./get_logs.sh bots.txt

Either create CSV files or visual representations of each NN with:

1.	Get the .json file you want
2.	Put it in the same directory as `export_data.py` or
`neural_net_vis.py`

    python export_data FILE.json 

## Quick Setup

These steps summarise how to start experiments during the PPSN 2016 tutorial:

1.  Connect to the conference WiFi
2.  Open a terminal (T1) and type 

    git clone https://github.com/ci-group/NEATThymio.git
    (for mac, install xcode-select --install  (error: xcrun error))

3.  Connect to ThymioNet (password: 172luckytulip75B)
4.  In T1 modify the bots file by correcting the IP address and send the files to the robot:

    nano NEATThymio/src/bots.txt 
    cd NEATThymio/src
    sh ./sync_all.sh 
    type “yes” and use the password raspberry (multiple times)

5.  Open a new terminal (T2) and login to the robot

    ssh pi@192.168.1.y (where y= pi address)
    password: raspberry

6.  Start an experiment with the command:
    sh ./start_one.sh obstacle_avoidance.py 192.168.1.y GO

7. To stop the robot use T1 and type 
    
    ./stop_all.sh

8. For passwordless access to the raspberry do:   

    ssh-keygen -t rsa
    ssh-copy-id pi@192.168.1.y



## Authors

- Tjarco Kerssens
- Cees Schouten
- Jacqueline Heinerman
