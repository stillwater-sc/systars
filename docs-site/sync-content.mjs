#!/usr/bin/env node
/**
 * Generates Starlight's src/content/docs/ from the repository's docs.
 *
 * Every page is authored outside docs-site/ and src/content/docs/ is entirely
 * generated (and git-ignored). Sources:
 *
 *   1. docs/**\/*.md     plain Markdown. The first H1 becomes the title; links
 *                       and image paths are rewritten for the site.
 *                       Pages in a subdirectory keep their path (design/,
 *                       architecture/, plan/, sessions/); pages at the top of
 *                       docs/ are placed by ROOT_PAGES.
 *   2. docs/site/       Starlight-native pages (the landing page), copied verbatim.
 *   3. ROOT_FILES       files from the repository root (CHANGELOG.md).
 *
 * Assets under docs/ (images, ...) are copied to public/assets/.
 *
 * A relative link that resolves to nothing fails the sync, and a top-level
 * page missing from ROOT_PAGES fails it too, so no page is silently dropped.
 * Anchors are checked after the build by starlight-links-validator.
 *
 * Run automatically by `npm run build` and `npm run dev`. The dev server also
 * re-runs sync() whenever a source changes (the integration in astro.config.mjs).
 */

import { existsSync, mkdirSync, readFileSync, readdirSync, rmSync, rmdirSync, writeFileSync } from 'fs';
import { dirname, join, posix, relative, sep } from 'path';
import { fileURLToPath } from 'url';

export const REPO = join(import.meta.dirname, '..');
export const DOCS = join(REPO, 'docs');
const SITE = join(DOCS, 'site');
const OUT = join(import.meta.dirname, 'src', 'content', 'docs');
const ASSETS = join(import.meta.dirname, 'public', 'assets');

const BASE = '/systars';
const GITHUB = 'https://github.com/stillwater-sc/systars';
const EDIT = `${GITHUB}/edit/main`;
const BLOB = `${GITHUB}/blob/main`;

/** Pages at the top of docs/ -> their place in the site (relative to content/docs/). */
const ROOT_PAGES = {
  'initial-development-environment.md': 'getting-started/development-environment.md',
  'development-workflow.md': 'getting-started/development-workflow.md',
  'isa-instructions.md': 'guides/isa-instructions.md',
  'wavefront-animation.md': 'guides/wavefront-animation.md',
};

/** Files from the repository root -> their place in the site. */
export const ROOT_FILES = {
  'CHANGELOG.md': 'changelog.md',
};

/** Sidebar order inside a group (lower first); pages not listed sort after, by title. */
const ORDER = {
  'getting-started/development-environment.md': 1,
  'getting-started/development-workflow.md': 2,
  'guides/isa-instructions.md': 1,
  'guides/wavefront-animation.md': 2,
};

// -- Collect the pages -----------------------------------------------

/** A platform path with forward slashes, as links and URLs use. */
const toPosix = (p) => p.split(sep).join('/');

/** Every file under `dir`, recursively (absolute paths). */
function walk(dir) {
  const files = [];
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    const path = join(dir, entry.name);
    if (entry.isDirectory()) files.push(...walk(path));
    else files.push(path);
  }
  return files;
}

/** repo-relative source path -> destination path relative to content/docs/ */
let pages = {};
/** asset paths relative to docs/ */
let assets = [];
let errors = [];

/** Collect the pages and assets of the current docs/ tree. */
function collect() {
  pages = {};
  assets = [];
  errors = [];
  for (const file of walk(DOCS)) {
    const rel = toPosix(relative(DOCS, file));
    if (rel.startsWith('site/')) continue;
    if (!rel.endsWith('.md')) {
      assets.push(rel);
      continue;
    }
    if (!rel.includes('/')) {
      if (!(rel in ROOT_PAGES)) {
        errors.push(`docs/${rel}: a top-level page needs a place in ROOT_PAGES (sync-content.mjs)`);
        continue;
      }
      pages[`docs/${rel}`] = ROOT_PAGES[rel];
    } else {
      pages[`docs/${rel}`] = rel;
    }
  }
  for (const [src, dest] of Object.entries(ROOT_FILES)) pages[src] = dest;
}

/** The site URL of a page, from its path relative to content/docs/ (`guides/x.md` -> `/systars/guides/x/`). */
function pageUrl(dest) {
  const slug = dest.replace(/\.mdx?$/, '').replace(/(^|\/)index$/, '');
  return `${BASE}/${slug}${slug ? '/' : ''}`;
}

// -- Rewriting -------------------------------------------------------

/**
 * Rewrite the relative links and image paths of one page (outside code).
 * @param {string} body the Markdown
 * @param {string} src the page's repo-relative path
 */
function rewriteLinks(body, src) {
  const srcDir = posix.dirname(src);
  const rewrite = (text) =>
    text.replace(/\]\(([^)\s]+)(\s+"[^"]*")?\)/g, (match, target, title = '') => {
      if (/^([a-z][a-z0-9+.-]*:|#|\/)/i.test(target)) return match;
      const [path, anchor] = target.split('#');
      const resolved = posix.normalize(posix.join(srcDir, decodeURI(path)));
      const hash = anchor === undefined ? '' : `#${anchor}`;
      let url;
      if (resolved in pages) url = pageUrl(pages[resolved]) + hash;
      else if (resolved.startsWith('docs/') && assets.includes(resolved.slice(5)))
        url = `${BASE}/assets/${resolved.slice(5)}`;
      else if (!resolved.startsWith('..') && existsSync(join(REPO, resolved)))
        url = `${BLOB}/${resolved}${hash}`;
      else {
        errors.push(`${src}: link to ${target} resolves to nothing (${resolved})`);
        return match;
      }
      return `](${url}${title})`;
    });

  // Leave fenced code blocks and inline code alone
  return body
    .split(/(^```[\s\S]*?^```)/m)
    .map((part, i) => (i % 2 ? part : part.split(/(`[^`\n]*`)/).map((s, j) => (j % 2 ? s : rewrite(s))).join('')))
    .join('');
}

/** A double-quoted YAML scalar. */
const yamlString = (s) => `"${s.replace(/\\/g, '\\\\').replace(/"/g, '\\"')}"`;

/**
 * A docs/ page as a Starlight page: its H1 becomes the title, it gets an editUrl
 * to its source, session logs get a dated sidebar label (newest first), and its
 * links are rewritten.
 * @param {string} content the Markdown
 * @param {string} src the page's repo-relative path
 * @param {string} dest its path relative to content/docs/
 */
function frontmatter(content, src, dest) {
  const h1 = content.match(/^#\s+(.+)$/m);
  if (!h1) errors.push(`${src}: no H1 to use as the title`);
  const title = h1 ? h1[1].trim() : 'Untitled';
  const lines = [`title: ${yamlString(title)}`, `editUrl: ${yamlString(`${EDIT}/${src}`)}`];

  const sidebar = [];
  // Session logs: label them by date, newest first
  const session = dest.match(/^sessions\/(\d{4}-\d{2}-\d{2})-/);
  if (session) {
    const label = title.replace(/^Session( Log)?:\s*/, '');
    sidebar.push(`  label: ${yamlString(`${session[1]} ${label}`)}`);
    sidebar.push(`  order: ${-Number(session[1].replace(/-/g, ''))}`);
  } else if (dest in ORDER) {
    sidebar.push(`  order: ${ORDER[dest]}`);
  }
  if (sidebar.length) lines.push('sidebar:', ...sidebar);

  const body = h1 ? content.replace(h1[0], '').replace(/^\n+/, '') : content;
  return `---\n${lines.join('\n')}\n---\n\n${rewriteLinks(body, src)}`;
}

/** A site page keeps its frontmatter; add the editUrl unless it sets one. */
function addEditUrl(content, src) {
  const fm = content.match(/^---\r?\n([\s\S]*?)\r?\n---/);
  if (!fm || /^editUrl\s*:/m.test(fm[1])) return content;
  return content.replace(/^---\r?\n/, `---\neditUrl: ${yamlString(`${EDIT}/${src}`)}\n`);
}

// -- Main ------------------------------------------------------------

/** Write a file only when its content changes, so the dev server reloads only what changed. */
function write(out, content, written) {
  written.add(out);
  if (existsSync(out) && readFileSync(out).equals(Buffer.from(content))) return;
  mkdirSync(dirname(out), { recursive: true });
  writeFileSync(out, content);
}

/** Remove the files of a generated tree that this sync did not write, and empty directories. */
function prune(dir, written) {
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    const path = join(dir, entry.name);
    if (entry.isDirectory()) {
      prune(path, written);
      if (readdirSync(path).length === 0) rmdirSync(path);
    } else if (!written.has(path)) rmSync(path);
  }
}

/**
 * Generate src/content/docs/ and public/assets/ from the sources.
 * @returns {string[]} the errors (an unplaced page, a link that resolves to nothing)
 */
export function sync() {
  collect();
  const written = new Set();
  for (const dir of [OUT, ASSETS]) mkdirSync(dir, { recursive: true });

  for (const [src, dest] of Object.entries(pages)) {
    write(join(OUT, dest), frontmatter(readFileSync(join(REPO, src), 'utf-8'), src, dest), written);
  }
  if (existsSync(SITE)) {
    for (const file of walk(SITE)) {
      const rel = toPosix(relative(SITE, file));
      write(join(OUT, rel), addEditUrl(readFileSync(file, 'utf-8'), `docs/site/${rel}`), written);
    }
  }
  for (const rel of assets) write(join(ASSETS, rel), readFileSync(join(DOCS, rel)), written);

  prune(OUT, written);
  prune(ASSETS, written);
  for (const e of errors) console.error(`  ERROR: ${e}`);
  return errors;
}

if (process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1]) {
  // Clear Astro's stale data store before a full build or dev start
  const astroCache = join(import.meta.dirname, 'node_modules', '.astro');
  if (existsSync(astroCache)) rmSync(astroCache, { recursive: true });

  console.log('Syncing docs/ -> docs-site/src/content/docs/ ...');
  if (sync().length) process.exit(1);
  console.log(`  ${Object.keys(pages).length} pages, ${assets.length} assets. Done.`);
}
