import type { Config } from "tailwindcss";
import colors from 'tailwindcss/colors';
import defaultTheme from "tailwindcss/defaultTheme";

const config: Config = {
  content: ["./src/**/*.{astro,html,js,jsx,md,mdx,svelte,ts,tsx,vue}"],
  darkMode: "class",
  theme: {
    extend: {
      fontFamily: {
        sans: ["Inter", ...defaultTheme.fontFamily.sans],
        mono: ["'Roboto Mono'", ...defaultTheme.fontFamily.mono],
      },
      colors: {
        primary: colors.sky,
        gray: colors.neutral,
      },
    },
  },
  plugins: [],
};

export default config;
