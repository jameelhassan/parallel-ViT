sleep 6300  # Sleep for 1 hour and 45 minutes (6300 seconds)

while true; do
    # Check the number of available GPUs
    num_gpus=$(nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name --format=csv,noheader,nounits | grep python | wc -l)

    if [ "$num_gpus" -eq 0 ]; then
        echo "All 4 GPUs are available. Launching the jobs..."
        
        # Run the jobs with CUDA_VISIBLE_DEVICES set
        CUDA_VISIBLE_DEVICES=2,3 python train_DDP.py --epochs 30 --lr 0.0002 --batch_size 320 --setting "ddp" --world_size 2 &
        CUDA_VISIBLE_DEVICES=0,1 python train_quantization.py --epochs 30 --lr 0.0002 --batch_size 320 --setting "ddp" --world_size 2 &
        
        # Wait for the background jobs to finish
        wait

        # Break out of the loop after jobs are completed
        break
    else
        echo "Not all 4 GPUs are available. Waiting for 5 minutes..."
        sleep 100  # Sleep for 5 minutes (300 seconds)
    fi
done