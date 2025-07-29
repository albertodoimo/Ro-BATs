#!/bin/bash

USER="thymio"
PI_LIST="pi_list.txt"

echo "🔍 Checking Chrony sync status on Raspberry Pis..."
echo ""

# Read all IPs into an array (handles stdin issues)
mapfile -t PI_ARRAY < "$PI_LIST"

for IP in "${PI_ARRAY[@]}"; do
    echo "----- $IP -----"

    if ping -c 1 -W 1 "$IP" > /dev/null 2>&1; then
        echo "✅ $IP is reachable via ping."

        echo "🔐 Connecting via SSH..."
        SYNC_OUTPUT=$(ssh -o ConnectTimeout=3 -o BatchMode=no "${USER}@${IP}" "chronyc tracking" 2>/dev/null)

        if [[ $? -eq 0 && -n "$SYNC_OUTPUT" ]]; then
            echo "$SYNC_OUTPUT" | grep -E 'Reference ID|Stratum|Last offset|Leap status'
        else
            echo "⚠️ SSH or chronyc failed on $IP."
        fi
    else
        echo "❌ $IP is not reachable via ping."
    fi

    echo ""
done
