/**
 * Smooth scrolling for same-page hash links.
 *
 * VitePress' router intercepts every in-site click in the capture phase and
 * jumps instantly to same-page anchors unless the link is a heading anchor.
 * This module is evaluated when the theme is imported, before the router
 * installs its listener, so registering here (also in the capture phase) runs
 * first: it handles the click itself and marks it as default-prevented, which
 * makes the router skip it.
 */
import { getScrollOffset, inBrowser } from 'vitepress'

function onClick(e: MouseEvent) {
  if (e.defaultPrevented || e.button !== 0 || e.ctrlKey || e.metaKey || e.shiftKey || e.altKey) return
  if (!(e.target instanceof Element)) return
  const link = e.target.closest('a')
  // Heading anchors already scroll smoothly through the router.
  if (!link || link.hasAttribute('target') || link.classList.contains('header-anchor')) return
  const href = link.getAttribute('href')
  if (!href) return

  const url = new URL(href, link.baseURI)
  const current = new URL(location.href)
  if (url.origin !== current.origin || url.pathname !== current.pathname || url.search !== current.search || !url.hash) return

  let target: HTMLElement | null = null
  try {
    target = document.getElementById(decodeURIComponent(url.hash.slice(1)))
  } catch {
    return
  }
  if (!target) return

  e.preventDefault()
  if (url.hash !== current.hash) {
    history.pushState({}, '', url.href)
    window.dispatchEvent(new HashChangeEvent('hashchange', { oldURL: current.href, newURL: url.href }))
  }

  const padding = parseInt(getComputedStyle(target).paddingTop, 10) || 0
  const top = window.scrollY + target.getBoundingClientRect().top - getScrollOffset() + padding
  const reduceMotion = matchMedia('(prefers-reduced-motion: reduce)').matches
  window.scrollTo({ left: 0, top, behavior: reduceMotion ? 'instant' : 'smooth' })
  // Keep keyboard and screen-reader position in sync with the visual jump.
  target.focus({ preventScroll: true })
}

if (inBrowser) window.addEventListener('click', onClick, { capture: true })
