#!/bin/bash

USER="thymio"
PI_LIST="./tracking/pi_list.txt"
PASSWORD="thymio"
NEW_CHRONY_SERVER=$(hostname -I | awk '{print $1}')

echo "🔍 Checking Chrony sync status on Raspberry Pis..."
echo ""

# Read all IPs into an array (handles stdin issues)
mapfile -t PI_ARRAY < "$PI_LIST"

for IP in "${PI_ARRAY[@]}"; do
    echo "----- $IP -----"
    echo ""

    if ping -c 1 -W 1 "$IP" > /dev/null 2>&1; then
        echo "✅ $IP is reachable via ping."
        echo ""

        echo "📝 Updating Chrony server IP on $IP to $NEW_CHRONY_SERVER..."
        sshpass -p "$PASSWORD" ssh -o ConnectTimeout=3 -o BatchMode=no "${USER}@${IP}" \
            "sudo sed -i 's/^server .*/server $NEW_CHRONY_SERVER iburst minpoll 2 maxpoll 2 xleave/' /etc/chrony/chrony.conf" 2>/dev/null
        

        echo "🔄 Restarting chrony service on $IP..."
        echo ""
        sshpass -p "$PASSWORD" ssh -o BatchMode=no "${USER}@${IP}" "sudo systemctl restart chrony" 2>/dev/null
        
        echo "⏳ Waiting for chrony to stabilize..."
        echo ""
        sleep 3

        echo "🔍 Checking chrony tracking status..."
        echo ""
        sshpass -p "$PASSWORD" ssh -o BatchMode=no "${USER}@${IP}" "chronyc -n tracking" 2>/dev/null
        
    else
        echo "❌ $IP is not reachable via ping."
    fi

    echo ""
done
