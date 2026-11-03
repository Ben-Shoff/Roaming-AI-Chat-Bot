#!/usr/bin/env bash
# Deploy this machine's copy of Sparky to the DGX Spark on the robot and start
# roam mode there.
#
# Usage (from Git Bash, or any Linux/macOS shell):
#   ./deploy.sh            copy the code to the Spark, then run chatbot_roam.py
#   ./deploy.sh --no-run   copy the code only
#
# Override the target with SPARK=user@host and the remote folder (relative to
# the remote home) with SPARK_DIR. run.sh expects ~/workspace/sparky.
#
# ssh asks for the Spark's password on each connection — twice per deploy. To
# skip that, install an SSH key once:
#   ssh-copy-id rshoff@10.41.43.203
# The password is deliberately not stored here: this folder is a git repo.
#
# The roam script runs in the foreground over ssh: its output shows here, and
# Ctrl-C stops it (and the robot). If the connection drops, the script is
# killed too and the robot-side watchdog stops the drive base.
set -euo pipefail

SPARK="${SPARK:-rshoff@10.41.43.203}"
SPARK_DIR="${SPARK_DIR:-workspace/sparky}"
FILES=(chatbot_roam.py chatbot.py run.sh setup.sh setup2.sh requirements.txt README.md CLAUDE.md)
SSH_OPTS=(-o ConnectTimeout=10)

run=1
case "${1:-}" in
  --no-run) run=0 ;;
  "") ;;
  *) echo "Usage: $0 [--no-run]" >&2; exit 2 ;;
esac

cd "$(dirname "$0")"
for f in "${FILES[@]}"; do
  [[ -f $f ]] || { echo "Missing $f — run this from the sparky folder." >&2; exit 1; }
done

echo "Deploying to $SPARK:~/$SPARK_DIR ..."
# One connection: back up the files about to be replaced, unpack the new
# ones, and strip any Windows line endings (CRLF breaks bash scripts and
# shebangs on Linux). The .venv, camera calibration and logs are untouched.
tar -cf - "${FILES[@]}" | ssh "${SSH_OPTS[@]}" "$SPARK" "
  set -e
  mkdir -p ~/$SPARK_DIR && cd ~/$SPARK_DIR
  backup=.deploy-backup/\$(date +%Y%m%d-%H%M%S)
  mkdir -p \$backup
  for f in ${FILES[*]}; do
    if [ -f \"\$f\" ]; then cp -p \"\$f\" \$backup/; fi
  done
  tar -xf -
  sed -i 's/\r\$//' ${FILES[*]}
  chmod +x run.sh setup.sh setup2.sh chatbot_roam.py chatbot.py
  echo \"  Previous versions backed up to ~/$SPARK_DIR/\$backup\"
"
echo "Deployed."

if (( run )); then
  echo "Starting roam mode on $SPARK (Ctrl-C to stop) ..."
  # Stop any copy that's already running first — it would be holding the mic
  # and cameras. [c] keeps pkill from matching this ssh command's own shell.
  # Ctrl-C it first so it stops the robot cleanly; force-kill after 10 s.
  ssh -t "${SSH_OPTS[@]}" "$SPARK" "
    pat='python.*[c]hatbot_roam\.py'
    if pkill -INT -f \"\$pat\"; then
      echo '  Stopping the copy that was already running ...'
      for i in 1 2 3 4 5 6 7 8 9 10; do pgrep -f \"\$pat\" >/dev/null || break; sleep 1; done
      pkill -KILL -f \"\$pat\" || true
    fi
    cd ~/$SPARK_DIR && bash run.sh
  "
fi
