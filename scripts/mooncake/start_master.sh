#!/bin/bash
# Start Mooncake Master Service
#
# This script starts the Mooncake Master Service which coordinates
# distributed KV cache storage for Eagle3 hidden states.
#
# By default, it enables the built-in HTTP metadata server, so you don't
# need to run a separate etcd or metadata service.
#
# Usage:
#   ./start_master.sh [OPTIONS]
#
# Options:
#   --port PORT           Master service port (default: 50051)
#   --http-metadata-port  HTTP metadata server port (default: 8080)
#   --http-metadata-host  HTTP metadata server host (default: 0.0.0.0)
#   --disable-http-metadata  Disable built-in HTTP metadata server
#   --high-availability   Enable HA mode with etcd (requires etcd cluster)
#   --etcd-endpoints      Etcd endpoints for HA mode (semicolon-separated)
#
# Environment variables:
#   MOONCAKE_BUILD_DIR    Path to Mooncake build directory
#
# Example:
#   # Simple single-node setup (metadata server included):
#   ./start_master.sh --port 50051
#
#   # The metadata server will be available at http://localhost:8090/metadata

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PORT=${PORT:-50051}
HTTP_METADATA_PORT=${HTTP_METADATA_PORT:-8090}
HTTP_METADATA_HOST=${HTTP_METADATA_HOST:-0.0.0.0}
ENABLE_HTTP_METADATA=${ENABLE_HTTP_METADATA:-true}
HIGH_AVAILABILITY=${HIGH_AVAILABILITY:-false}
ETCD_ENDPOINTS=${ETCD_ENDPOINTS:-""}
MOONCAKE_BUILD_DIR=${MOONCAKE_BUILD_DIR:-""}

while [[ $# -gt 0 ]]; do
    case $1 in
        --port)
            PORT="$2"
            shift 2
            ;;
        --http-metadata-port)
            HTTP_METADATA_PORT="$2"
            shift 2
            ;;
        --http-metadata-host)
            HTTP_METADATA_HOST="$2"
            shift 2
            ;;
        --disable-http-metadata)
            ENABLE_HTTP_METADATA=false
            shift
            ;;
        --high-availability)
            HIGH_AVAILABILITY=true
            shift
            ;;
        --etcd-endpoints)
            ETCD_ENDPOINTS="$2"
            shift 2
            ;;
        --build-dir)
            MOONCAKE_BUILD_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

MASTER_CMD=""
if [ -n "$MOONCAKE_BUILD_DIR" ]; then
    MASTER_CMD="$MOONCAKE_BUILD_DIR/mooncake-store/src/mooncake_master"
elif command -v mooncake_master &> /dev/null; then
    MASTER_CMD="mooncake_master"
else
    echo "ERROR: mooncake_master not found."
    echo "Please either:"
    echo "  1. Set MOONCAKE_BUILD_DIR to point to the Mooncake build directory"
    echo "  2. Install Mooncake and ensure mooncake_master is in PATH"
    exit 1
fi

ARGS="--port=$PORT"

if [ "$ENABLE_HTTP_METADATA" = "true" ]; then
    ARGS="$ARGS --enable_http_metadata_server=true"
    ARGS="$ARGS --http_metadata_server_port=$HTTP_METADATA_PORT"
    ARGS="$ARGS --http_metadata_server_host=$HTTP_METADATA_HOST"
fi

if [ "$HIGH_AVAILABILITY" = "true" ]; then
    if [ -z "$ETCD_ENDPOINTS" ]; then
        echo "ERROR: --etcd-endpoints required for high availability mode"
        exit 1
    fi
    ARGS="$ARGS --enable-ha=true"
    ARGS="$ARGS --etcd-endpoints=$ETCD_ENDPOINTS"
fi

echo "Starting Mooncake Master Service..."
echo "  Master Port: $PORT"
if [ "$ENABLE_HTTP_METADATA" = "true" ]; then
    echo "  HTTP Metadata Server: enabled"
    echo "    - Host: $HTTP_METADATA_HOST"
    echo "    - Port: $HTTP_METADATA_PORT"
    echo "    - URL: http://$HTTP_METADATA_HOST:$HTTP_METADATA_PORT/metadata"
else
    echo "  HTTP Metadata Server: disabled (requires external etcd)"
fi
echo "  High Availability: $HIGH_AVAILABILITY"
echo ""

exec $MASTER_CMD $ARGS
