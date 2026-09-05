/**
 * Next 14 reads `next.config.js` only — a `next.config.ts` sitting beside it is loaded by
 * nothing, which is how `reactStrictMode: true` came to be set in a file the framework never
 * opened. The `.ts` one is deleted and its single setting lives here.
 *
 * @type {import('next').NextConfig}
 */
const nextConfig = {
  reactStrictMode: true,
};

module.exports = nextConfig;
