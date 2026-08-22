/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./src/**/*.{js,ts,jsx,tsx,mdx}"],
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        brand: {
          50: "#eef5ff",
          100: "#d9e8ff",
          200: "#bcdbff",
          300: "#8ec5ff",
          400: "#58a4ff",
          500: "#3b82f6",
          600: "#1d5eeb",
          700: "#1549d8",
          800: "#173daf",
          900: "#19378a",
        },
      },
    },
  },
  plugins: [],
};
