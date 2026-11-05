import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import starlightLinksValidator from 'starlight-links-validator';
import { join, sep } from 'path';
import { DOCS, REPO, ROOT_FILES, sync } from './sync-content.mjs';

// `npm run dev` syncs once at start; this re-runs the sync whenever a page is
// added, changed or removed in docs/ (or in CHANGELOG.md), so the dev server
// reloads with the edit instead of serving the content of the start.
const syncOnEdit = {
  name: 'systars:sync-on-edit',
  hooks: {
    'astro:server:setup': ({ server, logger }) => {
      const sources = [DOCS, ...Object.keys(ROOT_FILES).map((f) => join(REPO, f))];
      server.watcher.add(sources);
      const resync = (path) => {
        if (!sources.some((s) => path === s || path.startsWith(s + sep))) return;
        if (sync().length) logger.warn('sync-content.mjs reported errors (above)');
      };
      for (const event of ['add', 'change', 'unlink']) server.watcher.on(event, resync);
    },
  },
};

export default defineConfig({
  site: 'https://stillwater-sc.github.io',
  base: '/systars',
  integrations: [
    syncOnEdit,
    starlight({
      title: 'Systars',
      description:
        'Python-based RTL generators of parallel operators (systolic arrays, SIMT, SIMD, stencil machines) using Amaranth HDL',
      social: [
        {
          icon: 'github',
          label: 'GitHub',
          href: 'https://github.com/stillwater-sc/systars',
        },
      ],
      // Turns the edit link on. Every page sets its own editUrl in
      // sync-content.mjs, because Starlight would join this base with the
      // generated (git-ignored) src/content/docs/ path.
      editLink: {
        baseUrl: 'https://github.com/stillwater-sc/systars/edit/main/docs/',
      },
      customCss: ['./src/styles/custom.css'],
      // A broken internal link or anchor fails the build
      plugins: [starlightLinksValidator()],
      sidebar: [
        {
          label: 'Getting Started',
          items: [{ autogenerate: { directory: 'getting-started' } }],
        },
        {
          label: 'Guides',
          items: [{ autogenerate: { directory: 'guides' } }],
        },
        {
          label: 'Architecture',
          items: [{ autogenerate: { directory: 'architecture' } }],
        },
        {
          label: 'Design',
          items: [{ autogenerate: { directory: 'design' } }],
        },
        {
          label: 'Planning',
          items: [{ autogenerate: { directory: 'plan' } }],
        },
        {
          label: 'Session Logs',
          collapsed: true,
          items: [{ autogenerate: { directory: 'sessions' } }],
        },
        { label: 'Changelog', link: '/changelog/' },
        {
          label: 'Related',
          items: [
            { label: 'kpu-sim', link: 'https://github.com/stillwater-sc/kpu-sim' },
            { label: 'Universal Numbers Library', link: 'https://stillwater-sc.github.io/universal/' },
            { label: 'MTL5 linear algebra', link: 'https://stillwater-sc.github.io/mtl5/' },
          ],
        },
      ],
    }),
  ],
});
