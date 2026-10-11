import { useEffect, useRef } from 'react'

/**
 * Always-running, cursor-reactive "review network" animation for the auth page.
 *
 * - Full-viewport <canvas>, pointer-events: none, so it never blocks the form or the 3D scene.
 * - Nodes drift slowly, connect with nearby nodes, and are pulled toward the cursor.
 * - Fast cursor movement leaves a glowing trail and emits ripples.
 * - Touch / no-mouse devices get an automatic virtual cursor on a smooth path.
 * - prefers-reduced-motion: draws one static frame, no animation loop.
 * - Pauses when the tab is hidden; cleans up every listener and frame on unmount.
 */

type Node = { x: number; y: number; vx: number; vy: number; r: number; hue: 0 | 1 }
type Ripple = { x: number; y: number; age: number }
type Trail = { x: number; y: number; age: number }

const COLORS = ['99,102,241', '34,211,238'] // indigo, cyan
const LINK_DIST = 130
const CURSOR_RADIUS = 190
const MAX_RIPPLES = 6
const TRAIL_LIFE = 40

export function CursorNetworkBackground() {
  const canvasRef = useRef<HTMLCanvasElement | null>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const reduced = window.matchMedia?.('(prefers-reduced-motion: reduce)').matches ?? false
    const coarse = window.matchMedia?.('(pointer: coarse)').matches ?? false

    let w = 0
    let h = 0
    let dpr = 1
    let nodes: Node[] = []
    const ripples: Ripple[] = []
    const trail: Trail[] = []
    let raf = 0
    let running = false

    // pointer: target (raw) and smoothed position
    const mouse = { x: -9999, y: -9999, sx: -9999, sy: -9999, active: false, lastMove: 0, speed: 0 }
    let virtual = coarse // touch devices start with the virtual cursor
    let t = 0
    let lastRipple = 0

    const resize = () => {
      dpr = Math.min(window.devicePixelRatio || 1, 2)
      w = window.innerWidth
      h = window.innerHeight
      canvas.width = Math.floor(w * dpr)
      canvas.height = Math.floor(h * dpr)
      canvas.style.width = `${w}px`
      canvas.style.height = `${h}px`
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
      const count = Math.max(40, Math.min(150, Math.round((w * h) / 11000)))
      nodes = Array.from({ length: count }, () => ({
        x: Math.random() * w,
        y: Math.random() * h,
        vx: (Math.random() - 0.5) * 0.25,
        vy: (Math.random() - 0.5) * 0.25,
        r: 1 + Math.random() * 1.6,
        hue: Math.random() < 0.7 ? 0 : 1,
      }))
      if (reduced) draw()
    }

    const onMove = (e: PointerEvent) => {
      if (e.pointerType === 'touch') return
      virtual = false
      const dx = e.clientX - mouse.x
      const dy = e.clientY - mouse.y
      mouse.speed = mouse.active ? Math.hypot(dx, dy) : 0
      mouse.x = e.clientX
      mouse.y = e.clientY
      if (!mouse.active) {
        mouse.sx = mouse.x
        mouse.sy = mouse.y
      }
      mouse.active = true
      mouse.lastMove = performance.now()
      trail.push({ x: mouse.x, y: mouse.y, age: 0 })
      if (trail.length > 60) trail.shift()
      if (mouse.speed > 28 && mouse.lastMove - lastRipple > 220 && ripples.length < MAX_RIPPLES) {
        ripples.push({ x: mouse.x, y: mouse.y, age: 0 })
        lastRipple = mouse.lastMove
      }
    }
    const onLeave = () => {
      mouse.active = false
    }

    function step() {
      t += 0.012
      // virtual cursor: smooth Lissajous path (touch devices, or before the first mouse move)
      if (virtual || !mouse.active) {
        mouse.x = w * (0.5 + 0.32 * Math.sin(t * 0.9))
        mouse.y = h * (0.5 + 0.28 * Math.sin(t * 1.3 + 1))
        if (mouse.sx < -999) {
          mouse.sx = mouse.x
          mouse.sy = mouse.y
        }
      } else if (performance.now() - mouse.lastMove > 2500) {
        mouse.active = false // idle mouse hands control back to the ambient path
      }
      mouse.sx += (mouse.x - mouse.sx) * 0.12
      mouse.sy += (mouse.y - mouse.sy) * 0.12

      for (const n of nodes) {
        const dx = mouse.sx - n.x
        const dy = mouse.sy - n.y
        const d = Math.hypot(dx, dy)
        if (d < CURSOR_RADIUS && d > 1) {
          const f = (1 - d / CURSOR_RADIUS) * 0.045
          n.vx += (dx / d) * f
          n.vy += (dy / d) * f
        }
        n.vx *= 0.985
        n.vy *= 0.985
        // keep a gentle ambient drift so nothing freezes
        n.vx += (Math.random() - 0.5) * 0.01
        n.vy += (Math.random() - 0.5) * 0.01
        n.x += n.vx
        n.y += n.vy
        if (n.x < -10) n.x = w + 10
        else if (n.x > w + 10) n.x = -10
        if (n.y < -10) n.y = h + 10
        else if (n.y > h + 10) n.y = -10
      }
      for (const p of trail) p.age++
      while (trail.length && trail[0].age > TRAIL_LIFE) trail.shift()
      for (const r of ripples) r.age++
      for (let i = ripples.length - 1; i >= 0; i--) if (ripples[i].age > 60) ripples.splice(i, 1)
    }

    function draw() {
      ctx!.clearRect(0, 0, w, h)
      ctx!.globalCompositeOperation = 'lighter'

      // links
      for (let i = 0; i < nodes.length; i++) {
        const a = nodes[i]
        for (let j = i + 1; j < nodes.length; j++) {
          const b = nodes[j]
          const dx = a.x - b.x
          if (dx > LINK_DIST || dx < -LINK_DIST) continue
          const dy = a.y - b.y
          const d2 = dx * dx + dy * dy
          if (d2 > LINK_DIST * LINK_DIST) continue
          const d = Math.sqrt(d2)
          const mx = (a.x + b.x) / 2 - mouse.sx
          const my = (a.y + b.y) / 2 - mouse.sy
          const near = Math.max(0, 1 - Math.hypot(mx, my) / CURSOR_RADIUS)
          const alpha = (1 - d / LINK_DIST) * (0.16 + near * 0.55)
          ctx!.strokeStyle = `rgba(${COLORS[a.hue]},${alpha.toFixed(3)})`
          ctx!.lineWidth = 0.6 + near * 0.9
          ctx!.beginPath()
          ctx!.moveTo(a.x, a.y)
          ctx!.lineTo(b.x, b.y)
          ctx!.stroke()
        }
      }

      // nodes
      for (const n of nodes) {
        const near = Math.max(0, 1 - Math.hypot(n.x - mouse.sx, n.y - mouse.sy) / CURSOR_RADIUS)
        const rad = n.r + near * 2.2
        const g = ctx!.createRadialGradient(n.x, n.y, 0, n.x, n.y, rad * 4)
        g.addColorStop(0, `rgba(${COLORS[n.hue]},${(0.5 + near * 0.5).toFixed(2)})`)
        g.addColorStop(1, `rgba(${COLORS[n.hue]},0)`)
        ctx!.fillStyle = g
        ctx!.beginPath()
        ctx!.arc(n.x, n.y, rad * 4, 0, Math.PI * 2)
        ctx!.fill()
      }

      // cursor trail
      for (const p of trail) {
        const k = 1 - p.age / TRAIL_LIFE
        ctx!.fillStyle = `rgba(34,211,238,${(k * 0.35).toFixed(3)})`
        ctx!.beginPath()
        ctx!.arc(p.x, p.y, 2 + k * 5, 0, Math.PI * 2)
        ctx!.fill()
      }

      // ripples
      for (const r of ripples) {
        const k = r.age / 60
        ctx!.strokeStyle = `rgba(99,102,241,${((1 - k) * 0.5).toFixed(3)})`
        ctx!.lineWidth = 1.5
        ctx!.beginPath()
        ctx!.arc(r.x, r.y, 10 + k * 110, 0, Math.PI * 2)
        ctx!.stroke()
      }

      // soft glow following the (smoothed) cursor; the native cursor is left untouched
      const glow = ctx!.createRadialGradient(mouse.sx, mouse.sy, 0, mouse.sx, mouse.sy, 140)
      glow.addColorStop(0, 'rgba(99,102,241,0.18)')
      glow.addColorStop(1, 'rgba(99,102,241,0)')
      ctx!.fillStyle = glow
      ctx!.beginPath()
      ctx!.arc(mouse.sx, mouse.sy, 140, 0, Math.PI * 2)
      ctx!.fill()

      ctx!.globalCompositeOperation = 'source-over'
    }

    const loop = () => {
      step()
      draw()
      raf = requestAnimationFrame(loop)
    }
    const start = () => {
      if (running || reduced) return
      running = true
      raf = requestAnimationFrame(loop)
    }
    const stop = () => {
      running = false
      cancelAnimationFrame(raf)
    }
    const onVisibility = () => (document.hidden ? stop() : start())

    resize()
    window.addEventListener('resize', resize)
    window.addEventListener('pointermove', onMove, { passive: true })
    document.addEventListener('mouseleave', onLeave)
    document.addEventListener('visibilitychange', onVisibility)
    start()

    return () => {
      stop()
      window.removeEventListener('resize', resize)
      window.removeEventListener('pointermove', onMove)
      document.removeEventListener('mouseleave', onLeave)
      document.removeEventListener('visibilitychange', onVisibility)
    }
  }, [])

  return (
    <canvas
      ref={canvasRef}
      aria-hidden="true"
      className="pointer-events-none fixed inset-0 z-[5] h-full w-full opacity-90"
    />
  )
}
