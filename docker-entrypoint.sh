#!/bin/bash
set -e

# Create OMEGA_HOME if needed
mkdir -p "${OMEGA_HOME:-/data/omega}"

# Run database migrations if DATABASE_URL is set
if [ -n "$DATABASE_URL" ]; then
    echo "omega: Running database migrations..."
    python -c "
from omega.db.migrate import migrate
migrate('$DATABASE_URL')
print('omega: Migrations complete.')
" 2>&1 || echo "omega: Migration warning (may already be up to date)"

    # Write secrets.json for cloud backend factory
    SECRETS_FILE="${OMEGA_HOME:-/data/omega}/secrets.json"
    if [ ! -f "$SECRETS_FILE" ]; then
        echo "{\"postgres_dsn\": \"$DATABASE_URL\"}" > "$SECRETS_FILE"
        chmod 600 "$SECRETS_FILE"
        echo "omega: Created $SECRETS_FILE"
    fi
fi

# Write config.json with user_id if not present
CONFIG_FILE="${OMEGA_HOME:-/data/omega}/config.json"
if [ ! -f "$CONFIG_FILE" ]; then
    USER_ID=$(python -c "import uuid; print(uuid.uuid4())")
    echo "{\"user_id\": \"$USER_ID\"}" > "$CONFIG_FILE"
    echo "omega: Generated user_id $USER_ID"
fi

echo "omega: Starting server..."
exec "$@"
