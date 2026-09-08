# Shared settings for the two-node MoE run (source me). Node roles:
#   CAPTURE_HOST (.103): 6 TP1 capture servers on GPUs 0-5, 100 GiB Mooncake segment each.
#   TRAINER_HOST (.102): mooncake_master, producer, 8-rank consumer.
CK=/personal/SpecForge-qwen38-moe
PY=/opt/sglang/bin/python
CAPTURE_HOST=10.13.114.103
TRAINER_HOST=10.13.114.102
N_CAPTURE=${N_CAPTURE:-6}
RECIPE=${1:?usage: $0 <recipe.yaml>}
export PYTHONPATH=/opt/sglang-0518:$CK MC_TRANSFER_TIMEOUT=300 SGLANG_SPEC_CAPTURE_SINK_CLIENTS=4
eval "$($PY - "$RECIPE" <<'PYEOF'
import sys, yaml, shlex
c = yaml.safe_load(open(sys.argv[1])); d = c["deployment"]["disaggregated"]; m = c["model"]
print("RUN_ID=" + shlex.quote(c["run_id"])); print("OUT=" + shlex.quote(c["output_dir"]))
print("CONTROL=" + shlex.quote(d["control_dir"])); print("CSTATE=" + shlex.quote(d["consumer_state_dir"]))
print("CLIENT_BUF=" + str(d.get("client_buffer_size", 1073741824)))
print("PROTOCOL=" + shlex.quote(d.get("mooncake_protocol", "tcp")))
print("SERVER_URLS=" + shlex.quote(",".join(d["server_urls"])))
print("NPROC=" + str(c["deployment"]["trainer"]["nproc_per_node"]))
print("MODEL=" + shlex.quote(m["target_model_path"])); print("ATTN=" + shlex.quote(m.get("sglang_attention_backend", "triton")))
print("MEMFRAC=" + str(m.get("sglang_mem_fraction_static", 0.85))); print("CTX=" + str(m.get("sglang_context_length", c["data"]["max_length"] + 7)))
print("MAXREQ=" + str(m.get("sglang_max_running_requests", 2))); print("MAXTOK=" + str(m.get("sglang_max_total_tokens", 32768)))
PYEOF
)"
export MOONCAKE_METADATA_SERVER=http://$TRAINER_HOST:35880/metadata MOONCAKE_MASTER_SERVER_ADDR=$TRAINER_HOST:35551 MOONCAKE_PROTOCOL=$PROTOCOL \
       DISAGG_SERVER_URLS=$SERVER_URLS DISAGG_BACKEND=mooncake DISAGG_CLIENT_BUFFER_SIZE=$CLIENT_BUF DISAGG_CLIENT_SEGMENT_SIZE=0 \
       DISAGG_DB=$CSTATE/consumer.sqlite DISAGG_INBOX_DIR=$CSTATE/inboxes DISAGG_REF_CHANNEL=$CONTROL/refs.jsonl DISAGG_STORE_ID=$RUN_ID
# Mooncake TCP transport tuning for cross-node feature reads (per-transfer throughput was ~14 MB/s with defaults):
export MC_SLICE_SIZE=${MC_SLICE_SIZE:-16777216} MC_RPC_CLIENT_IO_THREADS=${MC_RPC_CLIENT_IO_THREADS:-8}
LOG=$CONTROL/logs
