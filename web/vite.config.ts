import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

const backendTarget = process.env.VITE_BACKEND_TARGET || 'http://localhost:8000';

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    proxy: {
      '/api': {
        target: backendTarget,
        changeOrigin: true,
      },
      '/assets': {
        target: backendTarget,
        changeOrigin: true,
      },
      '/processed_ocr': {
        target: backendTarget,
        changeOrigin: true,
      },
    },
  },
});
