# =============================================================================
# MedCareAI Frontend Dockerfile
# Multi-stage build for production-ready image
# =============================================================================

FROM node:18-alpine as base

WORKDIR /app

# -----------------------------------------------------------------------------
# Development stage
# -----------------------------------------------------------------------------
FROM base as development

COPY package*.json ./
RUN npm install

COPY . .

EXPOSE 5173

CMD ["npm", "run", "dev", "--", "--host"]

# -----------------------------------------------------------------------------
# Build stage
# -----------------------------------------------------------------------------
FROM base as build

COPY package*.json ./
RUN npm ci

COPY . .
RUN npm run build

# -----------------------------------------------------------------------------
# Production stage
# -----------------------------------------------------------------------------
FROM nginx:alpine as production

COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
