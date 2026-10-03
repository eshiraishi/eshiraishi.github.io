export const DEFAULT_LOCALE = "pt-br" as const;
export const LOCALES = ["pt-br", "en"] as const;
export type Locale = (typeof LOCALES)[number];

export const LOCALE_LABELS: Record<Locale, { name: string; flag: string }> = {
  "pt-br": { name: "Português", flag: "🇧🇷" },
  en: { name: "English", flag: "🇺🇸" },
};

export const UI = {
  "pt-br": {
    posts: "Posts",
    contact: "Contato",
    selectLanguage: "Selecionar idioma",
    backToPosts: "Voltar para os posts",
    previous: "Anterior",
    next: "Próximo",
  },
  en: {
    posts: "Posts",
    contact: "Contact",
    selectLanguage: "Select language",
    backToPosts: "Back to posts",
    previous: "Previous",
    next: "Next",
  },
} as const satisfies Record<Locale, Record<string, string>>;

export function getPostSlug(post: {
  id: string;
  data: { translationKey?: string };
}) {
  return post.data.translationKey ?? post.id;
}
