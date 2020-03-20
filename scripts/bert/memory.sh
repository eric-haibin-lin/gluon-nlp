nvidia-smi --query-gpu=memory.used --format=csv | sed 's/\n/\t/g' | sed 's/Running/\nRunning/g' | sed 's/MiB/ /g' | tr '\n' '\t';
