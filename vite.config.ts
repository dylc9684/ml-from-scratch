import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  build: {
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (!id.includes("node_modules")) {
            return undefined;
          }

          if (id.includes("@tensorflow")) {
            return "vendor-tfjs";
          }

          if (id.includes("/d3") || id.includes("d3-")) {
            return "vendor-d3";
          }

          if (id.includes("/katex")) {
            return "vendor-katex";
          }

          if (id.includes("/prismjs")) {
            return "vendor-prism";
          }

          if (id.includes("/react") || id.includes("/react-dom") || id.includes("/scheduler")) {
            return "vendor-react";
          }

          return undefined;
        },
      },
    },
  },
});
