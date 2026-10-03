import { defineCollection } from "astro:content";
import { glob } from "astro/loaders";
import { z } from "astro/zod";

const posts = defineCollection({
  loader: glob({
    pattern: "**/*.{md,mdx}",
    base: "./src/content/posts",
    deferRender: true,
  }),
  schema: z.object({
    title: z.string(),
    description: z.string(),
    date: z.coerce.date(),
    draft: z.boolean().default(false),
    locale: z.enum(["pt-br", "en"]).default("pt-br"),
    translationKey: z.string().optional(),
    image: z.string().default("/blog-placeholder.png"),
    hideImage: z.boolean().default(true),
  }),
});

export const collections = { posts };
