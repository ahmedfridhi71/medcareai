/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // Beige palette (warm, soft)
        beige: {
          50: '#fbf8f3',
          100: '#f5efe2',
          200: '#ece0c4',
          300: '#dec99a',
          400: '#cdae6f',
          500: '#bf9852',
          600: '#a98146',
          700: '#88663b',
          800: '#705336',
          900: '#5d452f',
          950: '#322316',
        },
        // Black/charcoal palette
        ink: {
          50: '#f6f6f6',
          100: '#e7e7e7',
          200: '#d1d1d1',
          300: '#b0b0b0',
          400: '#888888',
          500: '#6d6d6d',
          600: '#5d5d5d',
          700: '#4f4f4f',
          800: '#2a2a2a',
          900: '#1a1a1a',
          950: '#0d0d0d',
        },
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        serif: ['Playfair Display', 'Georgia', 'serif'],
      },
      boxShadow: {
        soft: '0 2px 8px -2px rgba(26, 26, 26, 0.08)',
        warm: '0 8px 24px -8px rgba(168, 129, 70, 0.25)',
      },
    },
  },
  plugins: [],
}
