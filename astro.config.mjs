// @ts-check
import mdx from "@astrojs/mdx";
import sitemap from "@astrojs/sitemap";
import tailwind from "@astrojs/tailwind";
import { getCache } from "@beoe/cache";
import rehypeMermaid from "@beoe/rehype-mermaid";
import expressiveCode from "astro-expressive-code";
import { defineConfig } from "astro/config";
import rehypeAutolinkHeadings from "rehype-autolink-headings";
import rehypeKatex from "rehype-katex";
import rehypeSlug from "rehype-slug";
import remarkMath from "remark-math";
import remarkRehype from 'remark-rehype';
import { DARK_THEME, LIGHT_THEME, SITE_URL } from "./src/consts";

import icon from "astro-icon";

const cache = await getCache();
// @ts-check
const remarkMathConfig = { singleDollarTextMath: true, output: 'html', strict: true, trust: true };
const rehypeMermaidConfig = {
  strategy: "inline",
  darkScheme: "class",
  mermaidConfig: {
    theme: 'neutral',
    darkMode: true,
    logLevel: 'info'
  },
  cache
};
const expressiveCodeIntegration = expressiveCode({
  themes: [DARK_THEME, LIGHT_THEME],
  themeCssRoot: ':root',
  useDarkModeMediaQuery: false,
  useStyleReset: false,
  removeUnusedThemes: true,
  useThemedScrollbars: true,
  useThemedSelectionColors: true,
  themeCssSelector: (theme) => theme.name === DARK_THEME ? ':root.dark' : ':root:not(.dark)',
  styleOverrides: {
    uiFontFamily: "var(--font-sans), sans-serif",
    codeFontFamily: "var(--font-mono), monospace",
  },
})

export default defineConfig({
  site: SITE_URL,
  markdown: {
    remarkPlugins: [
      [remarkMath, remarkMathConfig],
      remarkRehype,
    ],
    rehypePlugins: [
      rehypeSlug,
      [rehypeAutolinkHeadings, {
        behavior: 'wrap',
        content: {
          type: 'element',
          tagName: 'img',
          properties: {
            src: '/link-icon.svg',
            alt: 'Link',
            className: ['link-icon']
          }
        },
        properties: {
          className: ['heading-anchor'],
          ariaLabel: 'Link to section'
        }
      }],
      [rehypeMermaid, rehypeMermaidConfig],
      rehypeKatex,
    ],
  },
  integrations: [expressiveCodeIntegration, mdx(), sitemap(), tailwind(), icon()],
});