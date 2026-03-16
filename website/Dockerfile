FROM node:22-slim AS builder

WORKDIR /app

COPY package.json package-lock.json* ./
RUN npm ci --legacy-peer-deps

COPY . .
RUN ASTRO_ADAPTER=node npm run build

FROM node:22-slim AS runtime

WORKDIR /app

# Copy built output
COPY --from=builder /app/dist ./dist
COPY --from=builder /app/node_modules ./node_modules
COPY --from=builder /app/package.json ./

ENV HOST=0.0.0.0 \
    PORT=3000 \
    NODE_ENV=production

EXPOSE 3000

# Astro Node adapter
CMD ["node", "dist/server/entry.mjs"]
