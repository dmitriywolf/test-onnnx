import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { viteStaticCopy } from "vite-plugin-static-copy";

// https://vite.dev/config/
export default defineConfig({
  // base: '/test-onnnx/',
  plugins: [
    react(),
    viteStaticCopy({
      targets: [
        {
          src: "src/models", // путь к файлу или папке, который нужно скопировать
          dest: "", // папка назначения в выходном каталоге
        },
      ],
    }),
  ],
  worker: {
    format: "es", // ES modules
  },
});
