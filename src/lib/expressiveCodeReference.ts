import { definePlugin } from "astro-expressive-code";
import { addClassName, h, setProperty } from "astro-expressive-code/hast";
import { getReferenceCodeMeta } from "./referenceCode";

declare module "@expressive-code/core" {
  interface ExpressiveCodeBlockProps {
    referenceTitle?: string;
    referenceSlug?: string;
  }
}

export function pluginReferenceCode() {
  return definePlugin({
    name: "Reference code blocks",
    hooks: {
      preprocessMetadata: ({ codeBlock }) => {
        const reference = getReferenceCodeMeta(codeBlock.meta);
        if (!reference) return;

        codeBlock.props.referenceTitle = reference.title;
        codeBlock.props.referenceSlug = reference.slug;
        codeBlock.props.frame = "none";
      },
      postprocessRenderedBlock: ({ codeBlock, renderData }) => {
        const title = codeBlock.props.referenceTitle;
        const slug = codeBlock.props.referenceSlug;
        if (!title || !slug) return;

        const block = renderData.blockAst;
        block.tagName = "details";
        addClassName(block, "reference-code");
        setProperty(block, "id", slug);
        setProperty(block, "data-reference-code", "true");

        block.children = block.children.filter(
          (child) => child.type !== "element" || child.tagName !== "figcaption",
        );

        block.children.unshift(
          h(
            "summary.reference-code-summary",
            { "data-reference-summary": "true" },
            [
              h(
                "svg.reference-code-icon",
                {
                  "aria-hidden": "true",
                  fill: "none",
                  focusable: "false",
                  viewBox: "0 0 24 24",
                  stroke: "currentColor",
                  "stroke-linecap": "round",
                  "stroke-linejoin": "round",
                  "stroke-width": "2",
                },
                [
                  h("path", {
                    d: "m18 16 4-4-4-4M6 8l-4 4 4 4m8.5-12-5 16",
                  }),
                ],
              ),
              h("span.reference-code-title", {}, [title]),
            ],
          ),
        );
      },
    },
  });
}
