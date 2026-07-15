export const config = {
  apiUrl: process.env.NEXT_PUBLIC_API_URL || "http://localhost:8080",
  websiteUrl: process.env.NEXT_PUBLIC_WEBSITE_URL || "https://github.com/ObedYAMEOGO",
} as const;