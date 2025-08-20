#!/bin/bash

ROBOT_IPS="./tracking/pi_list.txt"
USERNAME="thymio"
PASSWORD="thymio"

while IFS=',' read -r ip; do
    sshpass -p "$PASSWORD" ssh -o StrictHostKeyChecking=no "$USERNAME@$ip" << EOF
        pkill -f robatv1_2.py
EOF
    echo "✅ $ip Stopped"
done < "$ROBOT_IPS"
