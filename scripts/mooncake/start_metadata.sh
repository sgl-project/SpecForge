#!/bin/bash
# Start Mooncake Metadata Service (etcd)
#
# NOTE: This script is ONLY needed for advanced setups using etcd for high availability.
# For most use cases, use the built-in HTTP metadata server instead by running:
#
#   ./start_master.sh  # Automatically enables HTTP metadata server
#
# The built-in HTTP metadata server is simpler and doesn't require etcd.
# Only use etcd if you need high availability with multiple master nodes.
#
# Usage:
#   ./start_metadata.sh [OPTIONS]
#
# Options:
#   --data-dir DIR        Data directory for etcd (default: /tmp/mooncake-etcd)
#   --client-port PORT    Client port (default: 2379)
#   --peer-port PORT      Peer port (default: 2380)
#   --name NAME           Node name (default: mooncake-etcd)

set -e

DATA_DIR=${DATA_DIR:-/tmp/mooncake-etcd}
CLIENT_PORT=${CLIENT_PORT:-2379}
PEER_PORT=${PEER_PORT:-2380}
NODE_NAME=${NODE_NAME:-mooncake-etcd}
LISTEN_HOST=${LISTEN_HOST:-0.0.0.0}
ADVERTISE_HOST=${ADVERTISE_HOST:-localhost}

while [[ $# -gt 0 ]]; do
    case $1 in
        --data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --client-port)
            CLIENT_PORT="$2"
            shift 2
            ;;
        --peer-port)
            PEER_PORT="$2"
            shift 2
            ;;
        --name)
            NODE_NAME="$2"
            shift 2
            ;;
        --listen-host)
            LISTEN_HOST="$2"
            shift 2
            ;;
        --advertise-host)
            ADVERTISE_HOST="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

if ! command -v etcd &> /dev/null; then
    echo "ERROR: etcd not found. Please install etcd first."
    echo ""
    echo "Installation options:"
    echo "  Ubuntu/Debian: sudo apt-get install etcd"
    echo "  macOS: brew install etcd"
    echo "  From source: https://github.com/etcd-io/etcd/releases"
    exit 1
fi

mkdir -p "$DATA_DIR"

echo "Starting etcd metadata server..."
echo "  Data directory: $DATA_DIR"
echo "  Client endpoint: http://$ADVERTISE_HOST:$CLIENT_PORT"
echo "  Peer endpoint: http://$ADVERTISE_HOST:$PEER_PORT"
echo ""

exec etcd \
    --name "$NODE_NAME" \
    --data-dir "$DATA_DIR" \
    --listen-client-urls "http://$LISTEN_HOST:$CLIENT_PORT" \
    --advertise-client-urls "http://$ADVERTISE_HOST:$CLIENT_PORT" \
    --listen-peer-urls "http://$LISTEN_HOST:$PEER_PORT" \
    --initial-advertise-peer-urls "http://$ADVERTISE_HOST:$PEER_PORT" \
    --initial-cluster "$NODE_NAME=http://$ADVERTISE_HOST:$PEER_PORT"
