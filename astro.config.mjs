// @ts-check
import mdx from "@astrojs/mdx";
import { defineConfig } from "astro/config";

import sitemap from "@astrojs/sitemap";

import tailwind from "@astrojs/tailwind";
import { getCache } from "@beoe/cache";
import rehypeMermaid from "@beoe/rehype-mermaid";
import expressiveCode from "astro-expressive-code";
import rehypeAutolinkHeadings from 'rehype-autolink-headings';
import rehypeKatex from "rehype-katex";
import remarkMath from "remark-math";
import remarkRehype from 'remark-rehype';
import { DARK_THEME, LIGHT_THEME, SITE_URL } from "./src/consts";

const cache = await getCache();
// @ts-check

// https://astro.build/config
export default defineConfig({
  site: SITE_URL,
  markdown: {
    remarkPlugins: [
      // [remarkMermaid, { mermaidConfig: { theme: 'neutral' } }],
      [remarkMath, { singleDollarTextMath: true }],
      remarkRehype,
    ],
    rehypePlugins: [
      [rehypeMermaid, {
        strategy: "inline",
        darkScheme: "class",
        mermaidConfig: { theme: 'neutral' },
        cache,
      }],
      rehypeKatex,
      [rehypeAutolinkHeadings, { behavior: 'after' }],
    ],
  },
  integrations: [
    expressiveCode({
      themes: [DARK_THEME, LIGHT_THEME],
      themeCssRoot: ':root',
      themeCssSelector: (theme) =>
        theme.name === DARK_THEME ? ':root.dark' : ':root:not(.dark)',
      useDarkModeMediaQuery: false,
      useStyleReset: false,
      styleOverrides: {
        uiFontFamily: "var(--font-sans), sans-serif",
        codeFontFamily: "var(--font-mono), monospace",
        uiFontSize: '16px',
        codeFontSize: '16px'
      },
      removeUnusedThemes: true,
      useThemedScrollbars: true,
      useThemedSelectionColors: true,
      // plugins: []  ,
      // defaultProps: {
      // collapseStyle: 'collapsible-auto',
      // },
    }),
    mdx(), sitemap(), tailwind()],
});
