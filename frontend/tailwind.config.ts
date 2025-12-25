import type { Config } from "tailwindcss";

export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: "#0f1b1c",
        moss: "#2f4f3a",
        canary: "#f0c25a",
        clay: "#d7b896",
        mist: "#eef2f3"
      },
      fontFamily: {
        display: ["Fraunces", "serif"],
        body: ["Space Grotesk", "sans-serif"]
      },
      boxShadow: {
        glow: "0 0 40px rgba(240, 194, 90, 0.35)"
      }
    }
  },
  plugins: []
} satisfies Config;
