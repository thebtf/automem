#!/bin/bash
set -e

PUID=${PUID:-1001}
PGID=${PGID:-1001}

echo "[entrypoint] uid=$(id -u) → PUID=${PUID}, PGID=${PGID}"

# Remap group GID
current_gid=$(getent group appgroup | cut -d: -f3)
if [ "$current_gid" != "$PGID" ]; then
    groupmod -o -g "$PGID" appgroup
fi

# Remap user UID
current_uid=$(id -u appuser)
if [ "$current_uid" != "$PUID" ]; then
    usermod -o -u "$PUID" appuser
fi

# Fix ownership of writable runtime directories
chown appuser:appgroup /home/appuser

exec gosu appuser "$@"
