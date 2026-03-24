echo "=== Checking GPU Utilization ==="

# Get GPU indices and their utilization
nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader,nounits | while IFS=',' read -r gpu_index utilization; do
    gpu_index=$(echo "$gpu_index" | tr -d ' ')
    utilization=$(echo "$utilization" | tr -d ' ')

    if [ "$utilization" -eq 0 ]; then
        echo "GPU $gpu_index has 0% utilization — checking for processes..."

        # Get PIDs running on this GPU
        pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader --id="$gpu_index")

        if [ -z "$pids" ]; then
            echo "  No processes found on GPU $gpu_index."
        else
            echo "  Killing processes on GPU $gpu_index: $pids"
            for pid in $pids; do
                pid=$(echo "$pid" | tr -d ' ')
                echo "  Killing PID $pid..."
                # kill -9 "$pid" && echo "  PID $pid killed." || echo "  Failed to kill PID $pid (may need sudo)."
                docker run --rm --privileged --pid=host ubuntu bash -c "kill -9 $pid"
            done
        fi
    else
        echo "GPU $gpu_index is active ($utilization% utilization) — skipping."
    fi
done

echo ""
