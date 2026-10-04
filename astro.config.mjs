// @ts-check
import mdx from "@astrojs/mdx";
import { unified } from "@astrojs/markdown-remark";
import sitemap from "@astrojs/sitemap";
import tailwindcss from "@tailwindcss/vite";
import rehypeMermaid from "@beoe/rehype-mermaid";
import gruvboxDarkHard from "@shikijs/themes/gruvbox-dark-hard";
import gruvboxLightHard from "@shikijs/themes/gruvbox-light-hard";
import expressiveCode from "astro-expressive-code";
import { defineConfig } from "astro/config";
import rehypeAutolinkHeadings from "rehype-autolink-headings";
import rehypeMathjaxFira from "./src/lib/rehypeMathjaxFira";
import rehypeSlug from "rehype-slug";
import remarkMath from "remark-math";
import { DARK_THEME, SITE_URL } from "./src/consts";
import { pluginReferenceCode } from "./src/lib/expressiveCodeReference";

import icon from "astro-icon";

const cache = new Map();
// @ts-check
const remarkMathConfig = { singleDollarTextMath: true };
const rehypeRoundMermaidNodes = () => {
  /** @param {any} tree */
  const transform = (tree) => {
    /** @param {any} node */
    const visit = (node) => {
      if (node.type === "element" && node.tagName === "rect") {
        const className = node.properties?.className;
        const classNames = Array.isArray(className) ? className : [className];
        if (classNames.includes("text")) {
          node.properties.rx = 6;
          node.properties.ry = 6;
        }
      }
      node.children?.forEach(visit);
    };

    visit(tree);
  };

  return transform;
};
const rehypeMermaidConfig = {
  strategy: "inline",
  mermaidConfig: {
    theme: "base",
    darkMode: false,
    flowchart: {
      curve: "basis",
      htmlLabels: false,
      nodeSpacing: 32,
      rankSpacing: 36,
      padding: 8,
    },
    themeVariables: {
      fontFamily: '"Fira Code Variable", monospace',
      fontSize: "12px",
    },
    themeCSS: `
      .node,
      .cluster {
        color: var(--theme-code-surface);
      }

      .node rect,
      .node circle,
      .node ellipse,
      .node polygon,
      .node path {
        fill: currentColor;
        stroke: none;
        stroke-width: 0 !important;
      }

      .cluster rect {
        fill: currentColor;
        stroke: none;
      }

      .edgePath .path,
      .flowchart-link {
        fill: none;
        stroke: var(--mermaid-fg);
        stroke-width: 1.25px !important;
      }

      .marker,
      .marker path,
      .arrowMarkerPath {
        fill: var(--mermaid-fg) !important;
        stroke: var(--mermaid-fg) !important;
      }

      .label text,
      .label tspan,
      .nodeLabel,
      .edgeLabel,
      .edgeLabel p,
      .edgeLabel text {
        color: var(--mermaid-fg);
        fill: var(--mermaid-fg);
      }

      .edgeLabel rect,
      .edgeLabel .labelBkg {
        background: var(--mermaid-bg);
        fill: var(--mermaid-bg);
        opacity: 1;
        stroke: none;
      }
    `,
    logLevel: "info",
  },
  cache,
};
const expressiveCodeIntegration = expressiveCode({
  plugins: [pluginReferenceCode()],
  themes: [gruvboxDarkHard, gruvboxLightHard],
  themeCssRoot: ":root",
  useDarkModeMediaQuery: false,
  useStyleReset: false,
  removeUnusedThemes: true,
  useThemedScrollbars: true,
  useThemedSelectionColors: true,
  themeCssSelector: (theme) =>
    theme.name === DARK_THEME ? ":root.dark" : ":root:not(.dark)",
  styleOverrides: {
    uiFontFamily: "var(--font-sans), sans-serif",
    codeFontFamily: "var(--font-mono), monospace",
  },
});
const markdownProcessor = unified({
  remarkPlugins: [[remarkMath, remarkMathConfig]],
  rehypePlugins: [
    rehypeSlug,
    [
      rehypeAutolinkHeadings,
      {
        behavior: "wrap",
        content: {
          type: "element",
          tagName: "img",
          properties: {
            src: "/link-icon.svg",
            alt: "Link",
            className: ["link-icon"],
          },
        },
        properties: {
          className: ["heading-anchor"],
          ariaLabel: "Link to section",
        },
      },
    ],
    [rehypeMermaid, rehypeMermaidConfig],
    rehypeRoundMermaidNodes,
    rehypeMathjaxFira,
  ],
});

export default defineConfig({
  site: SITE_URL,
  i18n: {
    locales: ["pt-br", "en"],
    defaultLocale: "en",
    routing: {
      prefixDefaultLocale: false,
    },
  },
  markdown: {
    syntaxHighlight: { type: "shiki", excludeLangs: ["mermaid", "math"] },
    processor: markdownProcessor,
  },
  integrations: [expressiveCodeIntegration, mdx(), sitemap(), icon()],
  vite: {
    plugins: [tailwindcss()],
  },
});
