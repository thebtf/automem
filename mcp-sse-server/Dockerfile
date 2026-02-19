FROM node:18-slim

WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm ci --only=production

# Copy application code
COPY server.js ./

# Install gosu for PUID/PGID privilege-drop entrypoint
RUN apt-get update && apt-get install -y --no-install-recommends gosu \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user — UID/GID are remapped at runtime via PUID/PGID env vars
RUN groupadd --gid 1001 appgroup && \
    useradd --uid 1001 --gid appgroup --shell /bin/bash --create-home appuser

# Entrypoint handles PUID/PGID remapping and privilege drop
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Railway injects PORT automatically
EXPOSE 8080

ENTRYPOINT ["/entrypoint.sh"]
CMD ["node", "server.js"]
