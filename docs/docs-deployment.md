# Documentation Deployment (Mintlify)

The Agnitra docs are hosted on [Mintlify](https://www.mintlify.com/). This guide covers the project layout, how to preview changes, and how to trigger a production deploy.

## Project Structure

- `mint.json` — navigation, branding, and site metadata.
- `docs/` — Markdown pages referenced by the navigation config.
- `docs/index.md` — landing page rendered at `/`.
- Additional Markdown files are grouped in the sidebar using the `"navigation"` array inside `mint.json`.

## Local Preview

1. Install the Mintlify CLI (requires Node.js 18+):
   ```bash
   npm install -g mintlify
   ```
2. Run the dev server:
   ```bash
   mintlify dev
   ```
3. Open the printed URL (defaults to `http://localhost:3000`) to preview docs with hot reload.

## Publishing Workflow

1. Commit changes (including `mint.json` and any Markdown files).
2. Push to `main`. Mintlify listens for commits on `main` in the `agnitraai/agnitraai` repository.
3. Mintlify attempts to build using the root `mint.json`. You can monitor status at the [Mintlify dashboard](https://dashboard.mintlify.com/agnitraai/agnitraai).

If the deployment fails:

- Ensure `mint.json` exists at the repository root and validates against the schema (`https://mintlify.com/schema.json`).
- Confirm each slug in `"navigation"` points to a Markdown/MDX file inside `docs/` (without the `.md` extension).
- Check the activity log on the dashboard for stack traces.

## Tips

- Use descriptive `title` labels in the navigation to override generated sidebar names.
- Keep page filenames in `kebab-case` or `snake_case`; the slug is derived from the filename.
- You can embed React components with `.mdx` if richer layouts are required.
