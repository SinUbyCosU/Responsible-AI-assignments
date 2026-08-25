import paramiko
import os

IP = "216.128.144.102"
USER = "root"
PASS = "[8eE967Lg}!(GZoz"

files = [
    "color_biased_mnist.py", 
    "q2_background_bias.py", 
    "q3_long_tail.py"
]

print(f"Connecting to {USER}@{IP}...")
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

try:
    ssh.connect(IP, username=USER, password=PASS, timeout=10)
except Exception as e:
    print("Failed to connect:", e)
    exit(1)

print("Connected! Opening SFTP...")
sftp = ssh.open_sftp()
try:
    sftp.mkdir("resp_assignment")
except Exception:
    pass # probably already exists

for f in files:
    print(f"Uploading {f}...")
    sftp.put(f, f"resp_assignment/{f}")

sftp.close()

# The command installs dependencies and runs the 3 scripts sequentially.
# It ends with 'exec bash' so the tmux window stays open when the scripts finish.
tmux_command = "tmux new-session -d -s assignment 'cd resp_assignment && pip install torch torchvision numpy matplotlib pillow && echo \"Starting Q1\" && python color_biased_mnist.py && echo \"Starting Q2\" && python q2_background_bias.py && echo \"Starting Q3\" && python q3_long_tail.py; exec bash'"

print("Starting tmux session 'assignment'...")
stdin, stdout, stderr = ssh.exec_command(tmux_command)

err = stderr.read().decode()
if err:
    print("Stderr:", err)
else:
    print("Tmux session started successfully.")

ssh.close()
print("\nDone! You can ssh into the machine and type: tmux attach -t assignment")
