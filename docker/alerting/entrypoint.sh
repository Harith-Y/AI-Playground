#!/bin/sh
# Entrypoint script to substitute environment variables in alertmanager.yml

# Substitute environment variables in the config file
envsubst < /etc/alertmanager/alertmanager.yml.template > /etc/alertmanager/alertmanager.yml

# Start alertmanager with the processed config
exec /bin/alertmanager "$@"
