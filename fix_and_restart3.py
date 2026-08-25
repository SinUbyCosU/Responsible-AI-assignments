import paramiko

IP = '216.128.144.102'
USER = 'root'
PASS = '[8eE967Lg}!(GZoz'

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
try:
    ssh.connect(IP, username=USER, password=PASS, timeout=10)
    
    ssh.exec_command('tmux kill-session -t assignment')
    
    # We use the official PyTorch cu121 index to ensure compatibility and avoid the NCCL missing symbol error
    tmux_command = "tmux new-session -d -s assignment 'cd resp_assignment && pip install --break-system-packages --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu121 && pip install --break-system-packages numpy matplotlib pillow && echo \"Starting Q1\" && python3 color_biased_mnist.py && echo \"Starting Q2\" && python3 q2_background_bias.py && echo \"Starting Q3\" && python3 q3_long_tail.py; exec bash'"
    ssh.exec_command(tmux_command)
    
    ssh.close()
    print("Restarted with PyTorch cu121 wheels!")
except Exception as e:
    print("Error:", e)
