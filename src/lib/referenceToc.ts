import type { MarkdownHeading } from "astro";
import { getReferenceCodeMeta } from "./referenceCode";

export interface TocEntry extends MarkdownHeading {
  isReference?: boolean;
}

export function getTocEntries(
  body: string,
  headings: MarkdownHeading[],
): TocEntry[] {
  const entries: TocEntry[] = [];
  const lines = body.split(/\r?\n/);
  let headingIndex = 0;
  let fence: string | undefined;

  for (const line of lines) {
    const fenceMatch = line.match(/^\s*(`{3,}|~{3,})(.*)$/);
    if (fenceMatch) {
      const marker = fenceMatch[1];
      if (fence) {
        if (marker[0] === fence[0] && marker.length >= fence.length) {
          fence = undefined;
        }
      } else {
        const reference = getReferenceCodeMeta(fenceMatch[2]);
        if (reference) {
          entries.push({
            depth: 6,
            slug: reference.slug,
            text: reference.title,
            isReference: true,
          });
        }
        fence = marker;
      }
      continue;
    }

    if (fence) continue;

    const headingMatch = line.match(/^(#{1,6})\s+/);
    if (headingMatch) {
      const heading = headings[headingIndex++];
      const depth = headingMatch[1].length;
      if (heading && depth >= 2 && depth <= 3) entries.push(heading);
    }
  }

  return entries;
}
