#!/bin/bash

ROBOT_IPS="./tracking/pi_list.txt"
USERNAME="thymio"
PASSWORD="thymio"

while IFS=',' read -r ip; do
    echo "-----------Connecting to $ip to run robatv1_2.py script------------"
    
    sshpass -p "$PASSWORD" ssh -o StrictHostKeyChecking=no "$USERNAME@$ip" << 'EOF'
        if [ -f "/home/thymio/miniforge3/etc/profile.d/conda.sh" ]; then
            source "/home/thymio/miniforge3/etc/profile.d/conda.sh"
        elif [ -f "/home/thymio/miniforge-pypy3/etc/profile.d/conda.sh" ]; then
            source "/home/thymio/miniforge-pypy3/etc/profile.d/conda.sh"
        else
            echo "❌ Could not find conda.sh, check your installation"
            exit 1  
        fi
        
        conda activate robat
        echo "⏳ Conda environment 'robat' activated"
        sleep 2

        nohup python3 "robat_py/robat_swarm/passive_robat/Scripts/robatv1_2.py" >/dev/null 2>&1 &
        echo "✅ Started robatv1_2.py"
EOF

    echo "✅ Finished launching on $ip"
done < "$ROBOT_IPS"
