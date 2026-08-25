import paramiko

IP = '216.128.144.102'
USER = 'root'
PASS = '[8eE967Lg}!(GZoz'

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
try:
    ssh.connect(IP, username=USER, password=PASS, timeout=10)
    
    print("Uploading fixed q2_background_bias.py...")
    sftp = ssh.open_sftp()
    sftp.put('q2_background_bias.py', 'resp_assignment/q2_background_bias.py')
    sftp.close()

    print("Restarting tmux session from Q2...")
    ssh.exec_command('tmux kill-session -t assignment')
    
    # We skip Q1 since it already finished.
    tmux_command = "tmux new-session -d -s assignment 'cd resp_assignment && echo \"Starting Q2\" && python3 q2_background_bias.py && echo \"Starting Q3\" && python3 q3_long_tail.py; exec bash'"
    ssh.exec_command(tmux_command)
    
    ssh.close()
    print("Successfully restarted from Q2!")
except Exception as e:
    print("Error:", e)
