/**
 * Build-time list of every recipe under docs/recipes/, from its frontmatter.
 * Imported by RecipeIndex.vue; VitePress runs this only during the build.
 */
import { createContentLoader } from 'vitepress'

export interface Recipe {
  url: string
  title: string
  description: string
  target: string | null
  method: string | null
  topology: string | null
}

declare const data: Recipe[]
export { data }

export default createContentLoader('recipes/*.md', {
  transform(raw): Recipe[] {
    return raw
      .filter((page) => !/\/recipes\/(index\.html)?$/.test(page.url))
      .map(({ url, frontmatter }) => ({
        url,
        title: frontmatter.title ?? url.split('/').pop() ?? url,
        description: frontmatter.description ?? '',
        target: frontmatter.target ?? null,
        method: frontmatter.method ?? null,
        topology: frontmatter.topology ?? null,
      }))
      .sort((a, b) => a.title.localeCompare(b.title))
  },
})
