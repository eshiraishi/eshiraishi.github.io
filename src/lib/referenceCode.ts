export interface ReferenceCodeMeta {
  title: string;
  slug: string;
}

export function getReferenceCodeMeta(
  meta: string,
): ReferenceCodeMeta | undefined {
  const title = meta.match(
    /(?:^|\s)reference=(?:"([^"]+)"|'([^']+)'|([^\s]+))/i,
  );
  if (!title) return undefined;

  const value = title[1] ?? title[2] ?? title[3];
  if (!value) return undefined;

  const slug = value
    .normalize("NFKD")
    .replace(/[\u0300-\u036f]/g, "")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-|-$/g, "");

  return {
    title: value,
    slug: `reference-${slug}`,
  };
}
