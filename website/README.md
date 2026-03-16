# OMEGA Dashboard

Astro 6 + React admin dashboard for OMEGA Memory. Ported from the production Next.js dashboard.

## Architecture

- **Marketing pages**: Pure Astro (SSR) with Nav/Footer
- **Admin dashboard**: Single React island (`client:only="react"`) mounted at `/admin`
- **API routes**: Astro endpoints using Drizzle ORM
- **Auth**: PBKDF2 password hashing with cookie-based sessions
- **DB**: PostgreSQL (primary), Supabase, D1, or SQLite fallback

## Project Structure

```text
src/
  components/admin/     # 180 React components (production port)
    shell/              # Sidebar, TopBar, MobileNav, CommandPalette
    dashboard/          # KPI tiles, alerts, project cards
    coordination/       # Agent session flow visualization
    projects/           # Project overview, detail, architecture
    feed/               # Activity feed with filters
    insights/           # Heatmaps, radar, reports
    jobs/               # Schedule runs, approvals
    research/           # Findings, plans, scan
    memories/graph/     # 3D memory graph (Three.js)
    entities/graph/     # 3D entity graph
    skills/graph/       # 3D skills graph
    hooks/              # useSmartPoll, useAmbientStatus, etc.
    lib/                # types, constants, format, chartUtils
  layouts/Admin.astro   # Dark full-screen container (no Nav/Footer)
  pages/
    admin/index.astro   # Mounts AdminApp React island
    admin/login.astro   # Login form
    admin/memories/     # 3D graph pages
    api/admin/          # 58 API routes (Drizzle ORM)
  lib/
    auth/               # AuthEngine (PBKDF2, sessions)
    db/                 # Drizzle schema + getDb() factory
    api/                # requireAdminAuth helper
```

## Commands

| Command | Action |
|:--------|:-------|
| `npm install` | Install dependencies |
| `npm run dev` | Local dev server at localhost:4321 |
| `npm run build:docker` | Build for Docker (Node adapter) |
| `npm run admin:create -- --email X --password Y` | Create admin user |

## Docker

```sh
docker compose up -d
# Dashboard at http://localhost:3000
# Create first admin user:
docker exec <dashboard-container> node scripts/create-admin.mjs --email admin@example.com --password changeme
```

## Database Priority

1. Cloudflare D1 binding (Workers deployment)
2. PostgreSQL via `DATABASE_URL`
3. Supabase via `SUPABASE_DB_URL`
4. SQLite fallback via `OMEGA_DB_PATH`
