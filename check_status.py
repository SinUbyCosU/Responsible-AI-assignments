import paramiko

IP = '216.128.144.102'
USER = 'root'
PASS = '[8eE967Lg}!(GZoz'

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
try:
    ssh.connect(IP, username=USER, password=PASS, timeout=10)
    
    stdin, stdout, stderr = ssh.exec_command('tmux capture-pane -pt assignment | tail -n 25')
    output = stdout.read().decode('utf-8', 'ignore')
    
    stdin, stdout, stderr = ssh.exec_command('nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader')
    gpu_stats = stdout.read().decode('utf-8', 'ignore')
    
    with open('status.txt', 'w', encoding='utf-8') as f:
        f.write("TMUX LOGS:\n" + output.strip() + "\n\nGPU:\n" + gpu_stats.strip())
        
    ssh.close()
except Exception as e:
    with open('status.txt', 'w') as f:
        f.write(str(e))
