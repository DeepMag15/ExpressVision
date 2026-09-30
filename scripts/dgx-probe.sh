#!/usr/bin/env bash
# Probe a DGX before committing to a plan.
#
# Four things decide the setup, and guessing any of them wastes an afternoon:
# what GPUs there are, whether there is internet egress, whether jobs run
# directly or through a scheduler, and where there is room to write.
#
#   bash dgx-probe.sh
#
# Paste the output back and the plan can be made concrete.

echo "=================== HOST ==================="
hostname
uname -srm
if [ -f /etc/os-release ]; then . /etc/os-release; echo "os: $PRETTY_NAME"; fi
echo "cpus: $(nproc)   ram: $(free -g 2>/dev/null | awk '/^Mem:/{print $2" GB"}')"

echo
echo "=================== GPU ===================="
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,memory.total,compute_cap,driver_version \
             --format=csv,noheader 2>/dev/null
  echo "-- current load --"
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader 2>/dev/null
else
  echo "nvidia-smi NOT FOUND"
fi

echo
echo "================ SCHEDULER ================="
# A shared DGX usually wants jobs submitted, not run on the login node.
for c in sinfo squeue bsub qsub; do
  command -v $c >/dev/null 2>&1 && echo "found: $c"
done
command -v sinfo >/dev/null 2>&1 && sinfo -o "%P %a %l %D %G" 2>/dev/null | head -5
command -v sinfo >/dev/null 2>&1 || echo "no scheduler detected - likely run directly"

echo
echo "================= PYTHON =================="
for p in python3 python3.12 python3.11 uv conda; do
  command -v $p >/dev/null 2>&1 && echo "$p -> $($p --version 2>&1 | head -1)"
done
python3 -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.device_count(), 'devices')" 2>/dev/null \
  || echo "torch: not installed in default python"

echo
echo "================ NETWORK =================="
# The dataset pipeline pulls from Google Storage. Many university DGX boxes sit
# behind a proxy or have no egress at all, and that changes the whole approach.
for url in https://pypi.org/simple/ \
           https://storage.googleapis.com/public-datasets-lila/ \
           https://huggingface.co ; do
  code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 12 "$url" 2>/dev/null)
  printf "  %-58s %s\n" "$url" "${code:-TIMEOUT/BLOCKED}"
done
[ -n "$HTTP_PROXY$HTTPS_PROXY$http_proxy$https_proxy" ] \
  && echo "  proxy env is set: ${HTTPS_PROXY:-$https_proxy}" \
  || echo "  no proxy env set"

echo
echo "================= STORAGE ================="
echo "home:    $HOME"
df -h "$HOME" 2>/dev/null | tail -1
quota -s 2>/dev/null | tail -2
for d in /scratch /raid /data /mnt/data "$HOME/scratch"; do
  [ -d "$d" ] && printf "  %-16s %s\n" "$d" "$(df -h "$d" 2>/dev/null | tail -1)"
done

echo
echo "=================== DONE =================="
echo "The three answers that decide the plan:"
echo "  1. how many GPUs and how much VRAM each"
echo "  2. whether storage.googleapis.com returned 200 (dataset can be fetched here)"
echo "  3. whether a scheduler is present (submit jobs) or not (run directly)"
