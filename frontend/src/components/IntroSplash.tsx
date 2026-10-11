import { useEffect, useRef, useState, useCallback } from 'react'
import { motion, AnimatePresence } from 'motion/react'
import { ArrowRight, Sparkles } from 'lucide-react'

export interface IntroSplashProps {
  onDone: () => void
}

interface NodePoint {
  x: number
  y: number
  vx: number
  vy: number
  targetX: number
  targetY: number
  baseRadius: number
  color: string
  baseAlpha: number
  pulseOffset: number
  pulseSpeed: number
}

interface Ripple {
  x: number
  y: number
  radius: number
  maxRadius: number
  alpha: number
  color: string
}

interface TrailPoint {
  x: number
  y: number
  alpha: number
}

// Samples points from the text "InsightForge" rendered on an offscreen canvas
function sampleTextPoints(width: number, height: number, count: number): { x: number; y: number }[] {
  const offscreen = document.createElement('canvas')
  const ctx = offscreen.getContext('2d')
  const centerY = height * 0.46

  if (!ctx) {
    const fallback: { x: number; y: number }[] = []
    const radius = Math.min(width, height) * 0.22
    for (let i = 0; i < count; i++) {
      const angle = (i / count) * Math.PI * 2
      fallback.push({
        x: width / 2 + Math.cos(angle) * radius,
        y: centerY + Math.sin(angle) * (radius * 0.4),
      })
    }
    return fallback
  }

  // Calculate font size that fits comfortably across devices (360px mobile to 4K)
  const maxTextWidth = Math.min(width * 0.88, 900)
  // "InsightForge" is 12 chars; average char width in sans-serif bold is ~0.62 * fontSize
  let fontSize = Math.floor(maxTextWidth / 7.6)
  if (fontSize < 24) fontSize = 24
  if (fontSize > 72) fontSize = 72

  offscreen.width = Math.ceil(width)
  offscreen.height = Math.ceil(height)

  ctx.fillStyle = '#000000'
  ctx.fillRect(0, 0, offscreen.width, offscreen.height)

  ctx.font = `900 ${fontSize}px Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif`
  ctx.textAlign = 'center'
  ctx.textBaseline = 'middle'
  ctx.fillStyle = '#ffffff'
  ctx.fillText('InsightForge', width / 2, centerY)

  const imgData = ctx.getImageData(0, 0, offscreen.width, offscreen.height)
  const data = imgData.data

  const candidatePoints: { x: number; y: number }[] = []
  const step = fontSize > 48 ? 3 : 2

  for (let y = 0; y < offscreen.height; y += step) {
    for (let x = 0; x < offscreen.width; x += step) {
      const idx = (y * offscreen.width + x) * 4
      if (data[idx] > 140) {
        candidatePoints.push({ x, y })
      }
    }
  }

  if (candidatePoints.length === 0) {
    const fallback: { x: number; y: number }[] = []
    for (let i = 0; i < count; i++) {
      const angle = (i / count) * Math.PI * 2
      fallback.push({
        x: width / 2 + Math.cos(angle) * 160,
        y: centerY + Math.sin(angle) * 50,
      })
    }
    return fallback
  }

  // Sample evenly distributed points from the rendered letters
  const points: { x: number; y: number }[] = []
  const stepRatio = candidatePoints.length / count
  for (let i = 0; i < count; i++) {
    const index = Math.floor(i * stepRatio) % candidatePoints.length
    // Add tiny jitter so particles don't align on an unnatural pixel grid
    points.push({
      x: candidatePoints[index].x + (Math.random() - 0.5) * 1.5,
      y: candidatePoints[index].y + (Math.random() - 0.5) * 1.5,
    })
  }

  return points
}

const PALETTE = ['#22d3ee', '#818cf8', '#6366f1', '#38bdf8', '#ffffff']

export function IntroSplash({ onDone }: IntroSplashProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const animFrameRef = useRef<number | null>(null)

  // State for prefers-reduced-motion
  const [prefersReducedMotion, setPrefersReducedMotion] = useState<boolean>(() => {
    if (typeof window === 'undefined') return false
    return window.matchMedia('(prefers-reduced-motion: reduce)').matches
  })

  // Visual phase states
  const [showWordBackdrop, setShowWordBackdrop] = useState(false)
  const [showTagline, setShowTagline] = useState(false)
  const [isFadingOut, setIsFadingOut] = useState(false)
  const [isHoveringButton, setIsHoveringButton] = useState(false)

  // Parallax offsets (opposite to cursor)
  const [parallax, setParallax] = useState({ x: 0, y: 0 })

  // Custom cursor position and spring ring coordinates
  const [cursorPos, setCursorPos] = useState({ x: -100, y: -100 })
  const [ringPos, setRingPos] = useState({ x: -100, y: -100 })
  const [hasInteracted, setHasInteracted] = useState(false)

  // Mutable refs for high-frequency physics
  const pointerRef = useRef({
    x: -100,
    y: -100,
    prevX: -100,
    prevY: -100,
    speed: 0,
    hasRealMouse: false,
  })

  const ringSpringRef = useRef({ x: -100, y: -100, radius: 18, targetRadius: 18 })
  const nodesRef = useRef<NodePoint[]>([])
  const ripplesRef = useRef<Ripple[]>([])
  const trailRef = useRef<TrailPoint[]>([])
  const startTimeRef = useRef<number>(0)
  const doneCalledRef = useRef<boolean>(false)

  const triggerDone = useCallback(() => {
    if (doneCalledRef.current) return
    doneCalledRef.current = true
    setIsFadingOut(true)
    setTimeout(() => {
      onDone()
    }, 380)
  }, [onDone])

  // Detect prefers-reduced-motion changes
  useEffect(() => {
    const media = window.matchMedia('(prefers-reduced-motion: reduce)')
    const handler = (e: MediaQueryListEvent) => setPrefersReducedMotion(e.matches)
    media.addEventListener('change', handler)
    return () => media.removeEventListener('change', handler)
  }, [])

  // If user prefers reduced motion, show static presentation with simple fade
  useEffect(() => {
    if (prefersReducedMotion) {
      const timer = setTimeout(() => {
        triggerDone()
      }, 2400)
      return () => clearTimeout(timer)
    }
  }, [prefersReducedMotion, triggerDone])

  // Main canvas animation and physics engine
  useEffect(() => {
    if (prefersReducedMotion) return

    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    let width = window.innerWidth
    let height = window.innerHeight
    const dpr = Math.min(window.devicePixelRatio || 1, 2)

    function updateSize() {
      width = window.innerWidth
      height = window.innerHeight
      if (!canvas) return
      canvas.width = Math.floor(width * dpr)
      canvas.height = Math.floor(height * dpr)
      canvas.style.width = `${width}px`
      canvas.style.height = `${height}px`

      // Resample text points for new dimensions
      const textPoints = sampleTextPoints(width, height, nodesRef.current.length || 200)
      nodesRef.current.forEach((node, i) => {
        if (textPoints[i]) {
          node.targetX = textPoints[i].x
          node.targetY = textPoints[i].y
        }
      })
    }

    // Initialize 190 nodes (~150-250 glowing nodes)
    const NODE_COUNT = 195
    const initialPoints = sampleTextPoints(width, height, NODE_COUNT)
    const newNodes: NodePoint[] = []

    for (let i = 0; i < NODE_COUNT; i++) {
      const color = PALETTE[Math.floor(Math.random() * PALETTE.length)]
      newNodes.push({
        x: Math.random() * width,
        y: Math.random() * height,
        vx: (Math.random() - 0.5) * 0.7,
        vy: (Math.random() - 0.5) * 0.7,
        targetX: initialPoints[i] ? initialPoints[i].x : width / 2,
        targetY: initialPoints[i] ? initialPoints[i].y : height * 0.46,
        baseRadius: 1.4 + Math.random() * 1.6,
        color,
        baseAlpha: 0.35 + Math.random() * 0.55,
        pulseOffset: Math.random() * Math.PI * 2,
        pulseSpeed: 1.8 + Math.random() * 2.2,
      })
    }
    nodesRef.current = newNodes

    updateSize()
    window.addEventListener('resize', updateSize)

    // Pointer move listener
    function handlePointerMove(e: PointerEvent) {
      pointerRef.current.hasRealMouse = true
      setHasInteracted(true)

      const px = e.clientX
      const py = e.clientY

      const dx = px - (pointerRef.current.x === -100 ? px : pointerRef.current.x)
      const dy = py - (pointerRef.current.y === -100 ? py : pointerRef.current.y)
      const speed = Math.hypot(dx, dy)

      pointerRef.current.prevX = pointerRef.current.x
      pointerRef.current.prevY = pointerRef.current.y
      pointerRef.current.x = px
      pointerRef.current.y = py
      pointerRef.current.speed = speed

      setCursorPos({ x: px, y: py })

      // Subtle parallax: opposite to cursor (-12px to +12px)
      const pxRatio = px / width - 0.5
      const pyRatio = py / height - 0.5
      setParallax({
        x: -pxRatio * 16,
        y: -pyRatio * 12,
      })

      // Fast movement ripple trigger
      if (speed > 14) {
        ripplesRef.current.push({
          x: px,
          y: py,
          radius: 4,
          maxRadius: Math.min(speed * 2.2, 90),
          alpha: 0.65,
          color: '#22d3ee',
        })
        if (ripplesRef.current.length > 8) ripplesRef.current.shift()
      }

      // Add trail point
      trailRef.current.push({ x: px, y: py, alpha: 0.7 })
      if (trailRef.current.length > 12) trailRef.current.shift()
    }

    window.addEventListener('pointermove', handlePointerMove)

    startTimeRef.current = performance.now()
    let lastRippleTime = 0

    // Animation Loop
    function render(now: number) {
      if (!ctx || !canvas) return
      const elapsed = (now - startTimeRef.current) / 1000

      // Step 1: Handle Touch / Virtual Cursor Auto-Path if no real mouse moved
      let currentPointerX = pointerRef.current.x
      let currentPointerY = pointerRef.current.y

      if (!pointerRef.current.hasRealMouse) {
        const autoT = elapsed * 1.2
        const virtualX = width * 0.5 + Math.sin(autoT * 0.85) * (width * 0.28)
        const virtualY = height * 0.46 + Math.cos(autoT * 1.15) * (height * 0.16)
        currentPointerX = virtualX
        currentPointerY = virtualY

        setCursorPos({ x: virtualX, y: virtualY })

        const pxRatio = virtualX / width - 0.5
        const pyRatio = virtualY / height - 0.5
        setParallax({
          x: -pxRatio * 14,
          y: -pyRatio * 10,
        })

        // Periodic gentle ripple along virtual curve
        if (now - lastRippleTime > 320) {
          lastRippleTime = now
          ripplesRef.current.push({
            x: virtualX,
            y: virtualY,
            radius: 4,
            maxRadius: 55,
            alpha: 0.45,
            color: '#38bdf8',
          })
          if (ripplesRef.current.length > 8) ripplesRef.current.shift()
        }
      }

      // Spring ring follower
      const spring = ringSpringRef.current
      if (spring.x === -100) {
        spring.x = currentPointerX
        spring.y = currentPointerY
      } else {
        spring.x += (currentPointerX - spring.x) * 0.18
        spring.y += (currentPointerY - spring.y) * 0.18
      }
      setRingPos({ x: spring.x, y: spring.y })

      // Clear & Draw Navy Canvas Background (#0b1020)
      ctx.save()
      ctx.scale(dpr, dpr)

      ctx.fillStyle = '#0b1020'
      ctx.fillRect(0, 0, width, height)

      // Radial atmospheric glow in background center
      const bgGrad = ctx.createRadialGradient(
        width * 0.5,
        height * 0.46,
        10,
        width * 0.5,
        height * 0.46,
        Math.max(width, height) * 0.65
      )
      bgGrad.addColorStop(0, 'rgba(21, 28, 56, 0.75)')
      bgGrad.addColorStop(0.5, 'rgba(11, 16, 32, 0.95)')
      bgGrad.addColorStop(1, '#060913')
      ctx.fillStyle = bgGrad
      ctx.fillRect(0, 0, width, height)

      // SEQUENCE TIMINGS:
      // 0-0.8s: nodes fade in and connections draw themselves
      // 0.8-2.2s: nodes pull together to form "InsightForge", then relax
      // 2.2-3s: tagline "Turn reviews into decisions" fades in, with Skip button
      // 3.0s+: auto fade-out and trigger onDone()
      const fadeInProgress = Math.min(elapsed / 0.8, 1)

      let morphPullFactor = 0
      if (elapsed >= 0.8 && elapsed < 2.2) {
        const morphT = elapsed - 0.8 // 0 to 1.4s
        if (morphT < 0.85) {
          // Pulling together into "InsightForge"
          morphPullFactor = Math.min(morphT / 0.6, 1)
        } else {
          // Relaxing back outwards
          morphPullFactor = Math.max(0, 1 - (morphT - 0.85) / 0.55)
        }
      }

      // UI state triggers based on sequence timing
      if (elapsed >= 1.0 && !showWordBackdrop) {
        setShowWordBackdrop(true)
      }
      if (elapsed >= 2.2 && !showTagline) {
        setShowTagline(true)
      }
      if (elapsed >= 3.2 && !doneCalledRef.current) {
        triggerDone()
      }

      // Render ripples
      for (let i = ripplesRef.current.length - 1; i >= 0; i--) {
        const r = ripplesRef.current[i]
        r.radius += 2.4
        r.alpha *= 0.93
        if (r.alpha < 0.02 || r.radius >= r.maxRadius) {
          ripplesRef.current.splice(i, 1)
          continue
        }

        ctx.save()
        ctx.beginPath()
        ctx.arc(r.x, r.y, r.radius, 0, Math.PI * 2)
        ctx.strokeStyle = `rgba(34, 211, 238, ${r.alpha.toFixed(3)})`
        ctx.lineWidth = 1.6
        ctx.shadowColor = '#22d3ee'
        ctx.shadowBlur = 10
        ctx.stroke()
        ctx.restore()
      }

      // Render glowing mouse trail
      for (let i = trailRef.current.length - 1; i >= 0; i--) {
        const p = trailRef.current[i]
        p.alpha *= 0.88
        if (p.alpha < 0.03) {
          trailRef.current.splice(i, 1)
          continue
        }
        ctx.save()
        ctx.beginPath()
        ctx.arc(p.x, p.y, 3, 0, Math.PI * 2)
        ctx.fillStyle = `rgba(99, 102, 241, ${p.alpha.toFixed(3)})`
        ctx.shadowColor = '#6366f1'
        ctx.shadowBlur = 8
        ctx.fill()
        ctx.restore()
      }

      // Update and draw network nodes & connections
      const nodes = nodesRef.current
      const lineDistThreshold = 95 * fadeInProgress

      // Node Physics
      for (let i = 0; i < nodes.length; i++) {
        const node = nodes[i]

        // Drift velocity
        node.x += node.vx
        node.y += node.vy

        // Wrap around canvas edges during drift
        if (node.x < -20) node.x = width + 20
        if (node.x > width + 20) node.x = -20
        if (node.y < -20) node.y = height + 20
        if (node.y > height + 20) node.y = -20

        // Attraction to mouse cursor
        const distToPointer = Math.hypot(node.x - currentPointerX, node.y - currentPointerY)
        if (distToPointer < 170 && distToPointer > 5) {
          const attractStrength = (1 - distToPointer / 170) * 0.7
          node.x += ((currentPointerX - node.x) / distToPointer) * attractStrength
          node.y += ((currentPointerY - node.y) / distToPointer) * attractStrength
        }

        // Letter Morph Attraction (InsightForge constellation)
        if (morphPullFactor > 0) {
          const dx = node.targetX - node.x
          const dy = node.targetY - node.y
          const pullRate = 0.12 * morphPullFactor
          node.x += dx * pullRate
          node.y += dy * pullRate
        }
      }

      // Draw Connections (Network Graph)
      for (let i = 0; i < nodes.length; i++) {
        const a = nodes[i]
        for (let j = i + 1; j < nodes.length; j++) {
          const b = nodes[j]
          const dist = Math.hypot(a.x - b.x, a.y - b.y)

          if (dist < lineDistThreshold) {
            const lineMidX = (a.x + b.x) * 0.5
            const lineMidY = (a.y + b.y) * 0.5
            const distToMouse = Math.hypot(lineMidX - currentPointerX, lineMidY - currentPointerY)

            // Brighten lines near the cursor
            const mouseProximity = Math.max(0, 1 - distToMouse / 150)
            const baseLineAlpha = (1 - dist / lineDistThreshold) * 0.22 * fadeInProgress
            const finalAlpha = Math.min(1, baseLineAlpha + mouseProximity * 0.5)

            ctx.beginPath()
            ctx.moveTo(a.x, a.y)
            ctx.lineTo(b.x, b.y)

            if (mouseProximity > 0.1) {
              ctx.strokeStyle = `rgba(34, 211, 238, ${finalAlpha.toFixed(3)})`
              ctx.lineWidth = 1.2
            } else {
              ctx.strokeStyle = `rgba(99, 102, 241, ${finalAlpha.toFixed(3)})`
              ctx.lineWidth = 0.7
            }
            ctx.stroke()
          }
        }
      }

      // Draw Glowing Nodes
      for (let i = 0; i < nodes.length; i++) {
        const node = nodes[i]
        const pulse = Math.sin(elapsed * node.pulseSpeed + node.pulseOffset) * 0.35 + 0.65
        const distToMouse = Math.hypot(node.x - currentPointerX, node.y - currentPointerY)
        const mouseGlowBoost = Math.max(0, 1 - distToMouse / 160) * 0.4
        const alpha = Math.min(1, (node.baseAlpha * pulse + mouseGlowBoost) * fadeInProgress)

        // Outer glow
        const glowRadius = (node.baseRadius * 3.2 + (morphPullFactor > 0.4 ? 2 : 0)) * pulse
        const grad = ctx.createRadialGradient(node.x, node.y, 0, node.x, node.y, glowRadius)
        grad.addColorStop(0, node.color)
        grad.addColorStop(1, 'rgba(11, 16, 32, 0)')

        ctx.save()
        ctx.fillStyle = grad
        ctx.globalAlpha = alpha * 0.8
        ctx.beginPath()
        ctx.arc(node.x, node.y, glowRadius, 0, Math.PI * 2)
        ctx.fill()

        // Core bright dot
        ctx.fillStyle = node.color
        ctx.globalAlpha = alpha
        ctx.shadowColor = node.color
        ctx.shadowBlur = 8
        ctx.beginPath()
        ctx.arc(node.x, node.y, node.baseRadius, 0, Math.PI * 2)
        ctx.fill()
        ctx.restore()
      }

      ctx.restore()
      animFrameRef.current = requestAnimationFrame(render)
    }

    animFrameRef.current = requestAnimationFrame(render)

    return () => {
      window.removeEventListener('resize', updateSize)
      window.removeEventListener('pointermove', handlePointerMove)
      if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current)
    }
  }, [prefersReducedMotion, showTagline, showWordBackdrop, triggerDone])

  // Reduced motion alternative screen
  if (prefersReducedMotion) {
    return (
      <div className="fixed inset 0 z-50 flex items-center justify-center bg-[#0b1020] text-white p-6">
        <div className="max-w-md w-full text-center space-y-5">
          <div className="mx-auto flex size-14 items-center justify-center rounded-2xl bg-gradient-to-tr from-indigo-600 via-indigo-500 to-cyan-400 text-2xl font-black text-white shadow-xl shadow-indigo-500/30">
            I
          </div>
          <div className="space-y-1.5">
            <h1 className="text-3xl font-black tracking-tight text-white">InsightForge</h1>
            <p className="text-sm font-medium text-slate-300">Turn reviews into decisions</p>
          </div>
          <div className="pt-4">
            <button
              type="button"
              onClick={triggerDone}
              className="rounded-full border border-slate-700 bg-slate-900/90 px-6 py-2.5 text-xs font-bold text-white transition hover:border-cyan-400 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-cyan-400"
            >
              Enter Workspace
            </button>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div
      className={`fixed inset-0 z-50 overflow-hidden bg-[#0b1020] select-none transition-opacity duration-300 ${
        isFadingOut ? 'opacity-0 pointer-events-none' : 'opacity-100'
      } cursor-none`}
    >
      {/* Full-screen 60fps Network Canvas */}
      <canvas ref={canvasRef} className="absolute inset-0 block h-full w-full pointer-events-none" />

      {/* Top Header Bar with Keyboard-Focusable Skip Button */}
      <header className="absolute top-0 left-0 right-0 z-20 flex items-center justify-between p-6 sm:p-8">
        <div className="flex items-center gap-2.5 text-xs font-semibold text-slate-400">
          <span className="size-2 rounded-full bg-cyan-400 animate-ping" />
          <span className="hidden sm:inline">Review Intelligence Graph</span>
        </div>

        <button
          type="button"
          tabIndex={0}
          onClick={triggerDone}
          onMouseEnter={() => setIsHoveringButton(true)}
          onMouseLeave={() => setIsHoveringButton(false)}
          className="group flex items-center gap-2 rounded-full border border-slate-700/80 bg-slate-900/85 px-4 py-2 text-xs font-semibold text-white shadow-lg backdrop-blur-md transition-all hover:border-cyan-400/80 hover:bg-slate-800/95 hover:shadow-cyan-500/20 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-cyan-400"
        >
          <span>Skip</span>
          <ArrowRight
            size={14}
            className="text-cyan-400 transition-transform duration-200 group-hover:translate-x-0.5"
          />
        </button>
      </header>

      {/* Central Parallax Brand Container */}
      <div
        className="absolute inset-0 z-10 flex flex-col items-center justify-center p-6 pointer-events-none transition-transform duration-75 ease-out"
        style={{
          transform: `translate3d(${parallax.x}px, ${parallax.y}px, 0)`,
        }}
      >
        <div className="flex flex-col items-center text-center max-w-xl mx-auto space-y-4 sm:space-y-5">
          {/* Brand Icon Emblem */}
          <motion.div
            initial={{ opacity: 0, scale: 0.8, y: -10 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            transition={{ duration: 0.7, ease: 'easeOut' }}
            className="relative flex size-14 sm:size-16 items-center justify-center rounded-2xl bg-gradient-to-tr from-indigo-600 via-indigo-500 to-cyan-400 text-2xl sm:text-3xl font-black text-white shadow-2xl shadow-indigo-500/40 border border-white/20"
          >
            <span>I</span>
            <div className="absolute -inset-1.5 rounded-2xl bg-gradient-to-r from-indigo-500/20 to-cyan-400/20 blur-sm -z-10" />
          </motion.div>

          {/* Brand Name Title with subtle luminous glow */}
          <div className="relative">
            <h1 className="text-3xl sm:text-5xl lg:text-6xl font-black tracking-tight text-white drop-shadow-[0_2px_18px_rgba(34,211,238,0.25)]">
              InsightForge
            </h1>
            <span className="absolute -top-1 -right-8 rounded-md bg-indigo-500/25 border border-indigo-400/40 px-1.5 py-0.5 text-[10px] font-bold text-indigo-300">
              v2.0
            </span>
          </div>

          {/* Sequence 2.2s - 3s: Tagline & Pill badge */}
          <AnimatePresence>
            {showTagline && (
              <motion.div
                initial={{ opacity: 0, y: 12 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -8 }}
                transition={{ duration: 0.5, ease: 'easeOut' }}
                className="space-y-3"
              >
                <p className="text-sm sm:text-base lg:text-lg font-medium text-slate-200 tracking-wide">
                  Turn reviews into decisions
                </p>

                <div className="inline-flex items-center gap-1.5 rounded-full border border-cyan-500/30 bg-cyan-950/40 px-3.5 py-1 text-xs text-cyan-300 backdrop-blur-sm">
                  <Sparkles size={12} className="text-cyan-400 animate-pulse" />
                  <span>AI Review Intelligence & Defect Analytics</span>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>

      {/* Custom Spring Cursor (Small Dot + Large Damped Ring) */}
      {(hasInteracted || cursorPos.x > 0) && (
        <div className="pointer-events-none fixed inset-0 z-30">
          {/* Inner core pointer dot */}
          <div
            className="fixed size-2 -translate-x-1/2 -translate-y-1/2 rounded-full bg-cyan-400 shadow-[0_0_12px_#22d3ee] pointer-events-none"
            style={{
              left: `${cursorPos.x}px`,
              top: `${cursorPos.y}px`,
            }}
          />

          {/* Larger spring follower ring (expands over buttons) */}
          <div
            className={`fixed -translate-x-1/2 -translate-y-1/2 rounded-full pointer-events-none transition-[width,height,border-color,background-color] duration-200 ease-out ${
              isHoveringButton
                ? 'size-14 border-2 border-cyan-400 bg-cyan-400/10 shadow-[0_0_20px_rgba(34,211,238,0.3)]'
                : 'size-9 border border-cyan-400/60'
            }`}
            style={{
              left: `${ringPos.x}px`,
              top: `${ringPos.y}px`,
            }}
          />
        </div>
      )}
    </div>
  )
}
