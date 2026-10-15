import type { Theme } from 'vitepress'
import DefaultTheme from 'vitepress/theme'
import Landing from './components/Landing.vue'
import ModelGallery from './components/ModelGallery.vue'
import BenchmarkDashboard from './components/dashboard/BenchmarkDashboard.vue'
import RecipeIndex from './components/RecipeIndex.vue'
import InstallSelector from './components/InstallSelector.vue'
import './custom.css'
import './smooth-anchors'

export default {
  extends: DefaultTheme,
  enhanceApp({ app }) {
    app.component('Landing', Landing)
    app.component('ModelGallery', ModelGallery)
    app.component('BenchmarkDashboard', BenchmarkDashboard)
    app.component('RecipeIndex', RecipeIndex)
    app.component('InstallSelector', InstallSelector)
  },
} satisfies Theme
