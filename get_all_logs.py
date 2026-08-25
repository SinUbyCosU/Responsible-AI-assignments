import paramiko

IP = '216.128.144.102'
USER = 'root'
PASS = '[8eE967Lg}!(GZoz'

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
try:
    ssh.connect(IP, username=USER, password=PASS, timeout=10)
    
    # -S - means capture from the very beginning of the scrollback buffer
    stdin, stdout, stderr = ssh.exec_command('tmux capture-pane -pt assignment -S -')
    output = stdout.read().decode('utf-8', 'ignore')
    
    with open('full_tmux_log.txt', 'w', encoding='utf-8') as f:
        f.write(output)
        
    ssh.close()
except Exception as e:
    with open('full_tmux_log.txt', 'w') as f:
        f.write(str(e))
