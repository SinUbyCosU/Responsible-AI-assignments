import paramiko

IP = '216.128.144.102'
USER = 'root'
PASS = '[8eE967Lg}!(GZoz'

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
try:
    ssh.connect(IP, username=USER, password=PASS, timeout=10)
    
    # Kill the old session that failed
    ssh.exec_command('tmux kill-session -t assignment')
    
    # Start a new one with --break-system-packages
    tmux_command = "tmux new-session -d -s assignment 'cd resp_assignment && pip install --break-system-packages torch torchvision numpy matplotlib pillow && echo \"Starting Q1\" && python color_biased_mnist.py && echo \"Starting Q2\" && python q2_background_bias.py && echo \"Starting Q3\" && python q3_long_tail.py; exec bash'"
    ssh.exec_command(tmux_command)
    
    ssh.close()
    print("Restarted with --break-system-packages!")
except Exception as e:
    print("Error:", e)
