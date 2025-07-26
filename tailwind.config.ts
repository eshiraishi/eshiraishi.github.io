import type { Config } from "tailwindcss";
import colors from 'tailwindcss/colors';
import defaultTheme from "tailwindcss/defaultTheme";

const config: Config = {
  content: ["./src/**/*.{astro,html,js,jsx,md,mdx,svelte,ts,tsx,vue}"],
  darkMode: "class",
  theme: {
    extend: {
      fontFamily: {
        serif: ['Atkinson Hyperlegible Next Variable', ...defaultTheme.fontFamily.serif],
        sans: ['Atkinson Hyperlegible Next Variable', ...defaultTheme.fontFamily.sans],
        mono: ['Atkinson Hyperlegible Mono Variable', ...defaultTheme.fontFamily.mono],
      },
      colors: {
        primary: colors.neutral,
        neutral: colors.neutral,
      }
    }
  },
  plugins: [],
};

export default config;
