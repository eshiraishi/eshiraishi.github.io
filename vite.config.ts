import dns from 'node:dns'
import { defineConfig } from 'vite'

dns.setDefaultResultOrder('verbatim')

export default defineConfig({
    server: {
        allowedHosts: true
    }
})