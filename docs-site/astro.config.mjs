import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import starlightLinksValidator from 'starlight-links-validator';

export default defineConfig({
  site: 'https://stillwater-sc.github.io',
  base: '/systars',
  integrations: [
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
