import type { Config } from "tailwindcss";
import colors from 'tailwindcss/colors';
import defaultTheme from "tailwindcss/defaultTheme";

const config: Config = {
  content: ["./src/**/*.{astro,html,js,jsx,md,mdx,svelte,ts,tsx,vue}"],
  darkMode: "class",
  theme: {
    extend: {
      fontFamily: {
        serif: ['var(--font-serif)', ...defaultTheme.fontFamily.sans],
        sans: ['var(--font-sans)', ...defaultTheme.fontFamily.sans],
        mono: ['var(--font-mono)', ...defaultTheme.fontFamily.mono],
      },
      colors: {
        primary: colors.stone,
        gray: colors.stone,
      }
    }
    // typography: ({ theme }: { theme: any }) => ({
    //   DEFAULT: {
    //     css: {}
    //   }
    // })
  },
  plugins: [],
};

export default config;
