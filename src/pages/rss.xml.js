import rss from "@astrojs/rss";
import { getCollection } from "astro:content";
import { SITE_TITLE, SITE_DESCRIPTION } from "../consts";
import { DEFAULT_LOCALE, getPostSlug } from "../lib/i18n";

export async function GET(context) {
  const posts = await getCollection(
    "posts",
    (entry) => !entry.data.draft && entry.data.locale === DEFAULT_LOCALE,
  );
  return rss({
    title: SITE_TITLE,
    description: SITE_DESCRIPTION,
    site: context.site,
    items: posts.map((post) => ({
      ...post.data,
      link: `/posts/${getPostSlug(post)}/`,
    })),
  });
}
