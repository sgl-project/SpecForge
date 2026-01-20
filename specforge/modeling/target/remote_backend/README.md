# Disagg Mode
Run master first
```
scripts/mooncake/start_master.sh
```

Run inference worker
```
HOSTNAME=<ip_address> MOONCAKE_PROTOCOL=rdma ./examples/run_remote_inference.sh
```

Then run training
```
./examples/run_remote_training.sh 
```

Run test with one:
```
python examples/send_remote_task.py 
```

Due to torch compile, it will be a bit slow at first.