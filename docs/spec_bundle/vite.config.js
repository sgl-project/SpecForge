import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

// https://vite.dev/config/
export default defineConfig({
  plugins: [vue()],
  base: './', // Use relative paths for deployment
  build: {
    outDir: 'dist',
    assetsDir: 'assets',
    sourcemap: false,
    minify: 'esbuild', // Use esbuild for faster minification (Vite built-in)
    rollupOptions: {
      output: {
        manualChunks: {
          'vue-vendor': ['vue'],
          'echarts-vendor': ['echarts', 'vue-echarts'],
          'csv-vendor': ['papaparse']
        }
      }
    }
  }
})
