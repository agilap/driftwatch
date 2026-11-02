#!/bin/sh
set -eu

echo "Applying DriftWatch database migrations"
alembic upgrade head
echo "Database migrations succeeded; starting application"
exec "$@"
