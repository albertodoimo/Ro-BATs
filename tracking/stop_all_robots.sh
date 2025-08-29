#!/bin/bash

ROBOT_IPS="./tracking/pi_list.txt"
USERNAME="thymio"
PASSWORD="thymio"

while IFS=',' read -r ip; do
    echo ""
    echo "-----------Connecting to $ip -----------"
    sshpass -p "$PASSWORD" ssh -o StrictHostKeyChecking=no "$USERNAME@$ip" << EOF
        # pkill -SIGINT -f robatv1_2.py
        # pkill -SIGINT -u thymio
        pkill -TERM -u thymio

EOF
    echo "✅ $ip Stopped"
done < "$ROBOT_IPS"
