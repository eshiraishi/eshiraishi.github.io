// @ts-check
import mdx from "@astrojs/mdx";
import { defineConfig } from "astro/config";

import sitemap from "@astrojs/sitemap";

import tailwind from "@astrojs/tailwind";
import { pluginCollapsibleSections } from '@expressive-code/plugin-collapsible-sections';
import { pluginLineNumbers } from '@expressive-code/plugin-line-numbers';
import expressiveCode from "astro-expressive-code";
import rehypeKatex from "rehype-katex";
import remarkMath from "remark-math";
import remarkMermaid from 'remark-mermaidjs';
import remarkRehype from 'remark-rehype';
import { SITE_URL } from "./src/consts";

// @ts-check

// https://astro.build/config
export default defineConfig({
  site: SITE_URL,
  markdown: {
    remarkPlugins: [
      [remarkMermaid, { mermaidConfig: { theme: 'dark' } }],
      [remarkMath, { singleDollarTextMath: true }],
      remarkRehype,
    ],
    rehypePlugins: [
      rehypeKatex
    ],
    shikiConfig: {
      themes: {
        light: "github-light",
        dark: "material-theme-palenight",
      },
      wrap: true
    }
  },
  integrations: [
    expressiveCode({
      themes: ['dark-plus', 'light-plus'],
      styleOverrides: {
        uiFontFamily: "'Inter', sans-serif",
        codeFontFamily: "'Roboto Mono', monospace",
      },
      removeUnusedThemes: true,
      useStyleReset: false,
      useThemedScrollbars: true,
      useThemedSelectionColors: true,
      plugins: [pluginCollapsibleSections(), pluginLineNumbers()],
      defaultProps: {
        collapseStyle: 'collapsible-auto',
      },
    }),
    mdx(), sitemap(), tailwind()],
});
