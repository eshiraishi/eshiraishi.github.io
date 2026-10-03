import "@mathjax/src/mjs/input/tex/ams/AmsConfiguration.js";
import "@mathjax/src/mjs/input/tex/base/BaseConfiguration.js";
import "@mathjax/src/mjs/input/tex/newcommand/NewcommandConfiguration.js";
import "@mathjax/src/mjs/input/tex/noundefined/NoUndefinedConfiguration.js";
import "@mathjax/src/mjs/util/asyncLoad/esm.js";
import { MathJaxFiraFont } from "@mathjax/mathjax-fira-font/mjs/svg.js";
import { mathjax } from "@mathjax/src/mjs/mathjax.js";
import { TeX } from "@mathjax/src/mjs/input/tex.js";
import { SVG } from "@mathjax/src/mjs/output/svg.js";
import { liteAdaptor } from "@mathjax/src/mjs/adaptors/liteAdaptor.js";
import { RegisterHTMLHandler } from "@mathjax/src/mjs/handlers/html.js";
import { fromHtml } from "hast-util-from-html";
import { toText } from "hast-util-to-text";
import type { Element, ElementContent, Root } from "hast";
import { visitParents } from "unist-util-visit-parents";
import type { VFile } from "vfile";

const emptyClasses: readonly unknown[] = [];
const adaptor = liteAdaptor();
RegisterHTMLHandler(adaptor);

const mathDocument = mathjax.document("", {
  InputJax: new TeX({
    packages: ["base", "ams", "newcommand", "noundefined"],
  }),
  OutputJax: new SVG({
    fontData: MathJaxFiraFont,
    fontCache: "local",
    blacker: 5,
    linebreaks: { inline: false },
  }),
});

type MathNode = {
  element: Element;
  parent: Root | Element;
  scope: Element;
  displayMode: boolean;
  parents: Array<Root | Element>;
};

export default function rehypeMathjaxFira() {
  return async function transform(tree: Root, file: VFile) {
    const nodes: MathNode[] = [];

    visitParents(tree, "element", (element, parents) => {
      const classes = Array.isArray(element.properties.className)
        ? element.properties.className
        : emptyClasses;
      const languageMath = classes.includes("language-math");
      const mathDisplay = classes.includes("math-display");
      const mathInline = classes.includes("math-inline");

      if (!languageMath && !mathDisplay && !mathInline) return;

      let parent = parents[parents.length - 1];
      let scope = element;
      let displayMode = mathDisplay;

      if (
        element.tagName === "code" &&
        languageMath &&
        parent?.type === "element" &&
        parent.tagName === "pre"
      ) {
        scope = parent;
        parent = parents[parents.length - 2];
        displayMode = true;
      }

      if (!parent) return;
      nodes.push({ element, parent, scope, displayMode, parents });
    });

    for (const { element, parent, scope, displayMode, parents } of nodes) {
      const value = toText(scope, { whitespace: "pre" });
      let result: ElementContent[];

      try {
        const node = await mathDocument.convertPromise(value, {
          display: displayMode,
        });
        result = fromHtml(adaptor.outerHTML(node), { fragment: true })
          .children as ElementContent[];
      } catch (error) {
        const cause = error instanceof Error ? error : new Error(String(error));
        file.message("Could not render math with MathJax", {
          ancestors: [...parents, element],
          cause,
          place: element.position,
          source: "rehype-mathjax-fira",
        });
        result = [
          {
            type: "element",
            tagName: "span",
            properties: {
              className: ["mathjax-error"],
              title: cause.message,
            },
            children: [{ type: "text", value }],
          },
        ];
      }

      const index = parent.children.indexOf(scope);
      if (index !== -1) parent.children.splice(index, 1, ...result);
    }

    if (nodes.length > 0) {
      const style = fromHtml(
        adaptor.outerHTML(mathDocument.outputJax.styleSheet(mathDocument)),
        { fragment: true },
      ).children[0] as ElementContent;
      tree.children.unshift(style);
    }
  };
}
