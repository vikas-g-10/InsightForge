import * as THREE from 'three'
import { useRef, useMemo, useState, useEffect, Component, type ReactNode } from 'react'
import { Canvas, useFrame, useThree } from '@react-three/fiber'
import { Html } from '@react-three/drei'
import { SceneFallback } from './SceneFallback'

// ----------------------------------------------------------------------
// WebGL Support Detection & Error Boundary
// ----------------------------------------------------------------------

function isWebGLAvailable(): boolean {
  if (typeof window === 'undefined') return false
  try {
    const canvas = document.createElement('canvas')
    return Boolean(
      window.WebGLRenderingContext &&
        (canvas.getContext('webgl2') || canvas.getContext('webgl') || canvas.getContext('experimental-webgl'))
    )
  } catch {
    return false
  }
}

interface ErrorBoundaryProps {
  children: ReactNode
  fallback: ReactNode
}

interface ErrorBoundaryState {
  hasError: boolean
}

class WebGLErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  constructor(props: ErrorBoundaryProps) {
    super(props)
    this.state = { hasError: false }
  }

  static getDerivedStateFromError() {
    return { hasError: true }
  }

  componentDidCatch(error: Error) {
    console.warn('WebGL Context / R3F error caught, rendering fallback:', error.message)
  }

  render() {
    if (this.state.hasError) {
      return this.props.fallback
    }
    return this.props.children
  }
}

// ----------------------------------------------------------------------
// Static Deterministic Geometry & Data
// ----------------------------------------------------------------------

function generateDustPositions(count: number): Float32Array {
  const pos = new Float32Array(count * 3)
  for (let i = 0; i < count; i++) {
    // Pure deterministic pseudo-random distribution
    const s1 = Math.sin(i * 127.1 + 311.7)
    const s2 = Math.cos(i * 269.5 + 183.3)
    const s3 = Math.sin(i * 419.2 + 371.9)
    pos[i * 3] = s1 * 7.5
    pos[i * 3 + 1] = s2 * 5.2
    pos[i * 3 + 2] = s3 * 4.0
  }
  return pos
}

const STATIC_DUST_POSITIONS = generateDustPositions(120)

// ----------------------------------------------------------------------
// 1. Central AI Insight Core ("The Insight")
// ----------------------------------------------------------------------

function InsightCore({ reducedMotion }: { reducedMotion: boolean }) {
  const coreRef = useRef<THREE.Group>(null)
  const innerSphereRef = useRef<THREE.Mesh>(null)
  const outerCageRef = useRef<THREE.Mesh>(null)
  const ring1Ref = useRef<THREE.Mesh>(null)
  const ring2Ref = useRef<THREE.Mesh>(null)

  useFrame(({ clock }) => {
    if (reducedMotion || document.hidden) return
    const t = clock.getElapsedTime()

    if (coreRef.current) {
      coreRef.current.rotation.y = t * 0.15
    }

    if (innerSphereRef.current) {
      const pulse = 1 + Math.sin(t * 2.2) * 0.05
      innerSphereRef.current.scale.set(pulse, pulse, pulse)
    }

    if (outerCageRef.current) {
      outerCageRef.current.rotation.x = t * 0.35
      outerCageRef.current.rotation.z = t * 0.2
    }

    if (ring1Ref.current) {
      ring1Ref.current.rotation.x = t * 0.5
      ring1Ref.current.rotation.y = t * 0.3
    }

    if (ring2Ref.current) {
      ring2Ref.current.rotation.y = -t * 0.4
      ring2Ref.current.rotation.z = t * 0.25
    }
  })

  return (
    <group ref={coreRef} position={[0, 0, 0]}>
      {/* Central Point Lights */}
      <pointLight color="#22d3ee" intensity={3.5} distance={9} decay={2} />
      <pointLight color="#a855f7" intensity={2.8} distance={8} decay={2} />

      {/* Glowing Inner Core */}
      <mesh ref={innerSphereRef}>
        <sphereGeometry args={[0.7, 32, 32]} />
        <meshStandardMaterial
          color="#06b6d4"
          emissive="#7c3aed"
          emissiveIntensity={1.4}
          roughness={0.2}
          metalness={0.8}
        />
      </mesh>

      {/* Outer Geodesic Lattice */}
      <mesh ref={outerCageRef}>
        <icosahedronGeometry args={[1.05, 1]} />
        <meshStandardMaterial
          color="#38bdf8"
          emissive="#0284c7"
          emissiveIntensity={0.8}
          wireframe
          transparent
          opacity={0.65}
        />
      </mesh>

      {/* Orbital Gyroscope Ring 1 (Cyan) */}
      <mesh ref={ring1Ref}>
        <torusGeometry args={[1.55, 0.02, 16, 64]} />
        <meshStandardMaterial
          color="#22d3ee"
          emissive="#06b6d4"
          emissiveIntensity={1.2}
          transparent
          opacity={0.75}
        />
      </mesh>

      {/* Orbital Gyroscope Ring 2 (Violet) */}
      <mesh ref={ring2Ref} rotation={[Math.PI / 3, 0, 0]}>
        <torusGeometry args={[1.8, 0.02, 16, 64]} />
        <meshStandardMaterial
          color="#c084fc"
          emissive="#9333ea"
          emissiveIntensity={1.2}
          transparent
          opacity={0.7}
        />
      </mesh>
    </group>
  )
}

// ----------------------------------------------------------------------
// 2. Positive Reviews ("The Good") - Glossy Emerald Bubbles & Chips
// ----------------------------------------------------------------------

interface ReviewItemProps {
  initialPos: [number, number, number]
  text: string
  stars: string
  aspect: string
  floatSpeed: number
  phaseOffset: number
  reducedMotion: boolean
}

function PositiveReviewBubble({
  initialPos,
  text,
  stars,
  aspect,
  floatSpeed,
  phaseOffset,
  reducedMotion,
}: ReviewItemProps) {
  const groupRef = useRef<THREE.Group>(null)
  const sphereRef = useRef<THREE.Mesh>(null)

  useFrame(({ clock }) => {
    if (reducedMotion || document.hidden || !groupRef.current) return
    const t = clock.getElapsedTime() * floatSpeed + phaseOffset

    // Gentle floating wobble
    groupRef.current.position.y = initialPos[1] + Math.sin(t) * 0.25
    groupRef.current.position.x = initialPos[0] + Math.cos(t * 0.7) * 0.12
    groupRef.current.position.z = initialPos[2] + Math.sin(t * 0.5) * 0.1

    // Gentle sphere breathing scale
    if (sphereRef.current) {
      const scale = 1 + Math.sin(t * 1.5) * 0.04
      sphereRef.current.scale.set(scale, scale, scale)
    }
  })

  return (
    <group ref={groupRef} position={initialPos}>
      {/* Glossy Emerald Glass Sphere */}
      <mesh ref={sphereRef}>
        <sphereGeometry args={[0.32, 28, 28]} />
        <meshStandardMaterial
          color="#10b981"
          emissive="#059669"
          emissiveIntensity={0.8}
          roughness={0.15}
          metalness={0.2}
          transparent
          opacity={0.85}
        />
      </mesh>

      {/* Orbiting Halo Ring */}
      <mesh rotation={[Math.PI / 4, 0, 0]}>
        <torusGeometry args={[0.42, 0.015, 12, 36]} />
        <meshBasicMaterial color="#34d399" transparent opacity={0.6} />
      </mesh>

      {/* Floating 2.5D Review Badge */}
      <Html position={[0, 0.48, 0]} center distanceFactor={8} zIndexRange={[10, 0]}>
        <div className="pointer-events-none select-none rounded-xl border border-emerald-500/40 bg-emerald-950/80 px-2.5 py-1 text-[11px] font-semibold text-emerald-200 shadow-lg shadow-emerald-950/60 backdrop-blur-md whitespace-nowrap flex items-center gap-1.5 transform hover:scale-105 transition-transform">
          <span className="text-emerald-400 font-bold">{stars}</span>
          <span className="text-white/90">"{text}"</span>
          <span className="rounded bg-emerald-500/20 px-1 py-0.2 text-[9px] text-emerald-300 uppercase tracking-wider">
            {aspect}
          </span>
        </div>
      </Html>
    </group>
  )
}

// ----------------------------------------------------------------------
// 3. Negative Reviews ("The Bad") - Coral / Crimson Cracked Cards
// ----------------------------------------------------------------------

function NegativeReviewCard({
  initialPos,
  text,
  stars,
  aspect,
  floatSpeed,
  phaseOffset,
  reducedMotion,
}: ReviewItemProps) {
  const groupRef = useRef<THREE.Group>(null)
  const cardRef = useRef<THREE.Mesh>(null)

  useFrame(({ clock }) => {
    if (reducedMotion || document.hidden || !groupRef.current) return
    const t = clock.getElapsedTime() * floatSpeed + phaseOffset

    // Micro impact tremor + floating motion
    const tremor = Math.sin(t * 8) * 0.015
    groupRef.current.position.y = initialPos[1] + Math.sin(t) * 0.22 + tremor
    groupRef.current.position.x = initialPos[0] + Math.sin(t * 0.6) * 0.12
    groupRef.current.position.z = initialPos[2] + Math.cos(t * 0.8) * 0.1

    if (cardRef.current) {
      cardRef.current.rotation.z = Math.sin(t * 0.5) * 0.08
    }
  })

  return (
    <group ref={groupRef} position={initialPos}>
      {/* Crimson Translucent Glass Box */}
      <mesh ref={cardRef}>
        <boxGeometry args={[0.5, 0.5, 0.1]} />
        <meshStandardMaterial
          color="#f43f5e"
          emissive="#e11d48"
          emissiveIntensity={0.7}
          roughness={0.25}
          metalness={0.3}
          transparent
          opacity={0.8}
        />
      </mesh>

      {/* Surface Crack Line Overlay */}
      <mesh position={[0, 0, 0.06]}>
        <ringGeometry args={[0.15, 0.18, 5]} />
        <meshBasicMaterial color="#fca5a5" wireframe transparent opacity={0.75} />
      </mesh>

      {/* Floating 2.5D Review Badge */}
      <Html position={[0, 0.48, 0]} center distanceFactor={8} zIndexRange={[10, 0]}>
        <div className="pointer-events-none select-none rounded-xl border border-rose-500/40 bg-rose-950/80 px-2.5 py-1 text-[11px] font-semibold text-rose-200 shadow-lg shadow-rose-950/60 backdrop-blur-md whitespace-nowrap flex items-center gap-1.5 transform hover:scale-105 transition-transform">
          <span className="text-rose-400 font-bold">{stars}</span>
          <span className="text-white/90">"{text}"</span>
          <span className="rounded bg-rose-500/20 px-1 py-0.2 text-[9px] text-rose-300 uppercase tracking-wider">
            {aspect}
          </span>
        </div>
      </Html>
    </group>
  )
}

// ----------------------------------------------------------------------
// 4. Data Streams into Core ("Conversion into Insight")
// ----------------------------------------------------------------------

function DataStreams({ reducedMotion }: { reducedMotion: boolean }) {
  const streamPoints = useMemo(() => {
    const streams: { curve: THREE.CatmullRomCurve3; color: string }[] = []

    // 2 Emerald streams from left
    streams.push({
      curve: new THREE.CatmullRomCurve3([
        new THREE.Vector3(-3.2, 1.8, -0.5),
        new THREE.Vector3(-1.8, 0.9, 0.2),
        new THREE.Vector3(-0.6, 0.2, 0),
        new THREE.Vector3(0, 0, 0),
      ]),
      color: '#10b981',
    })
    streams.push({
      curve: new THREE.CatmullRomCurve3([
        new THREE.Vector3(-2.8, -1.6, 0.3),
        new THREE.Vector3(-1.5, -0.8, -0.2),
        new THREE.Vector3(-0.5, -0.2, 0),
        new THREE.Vector3(0, 0, 0),
      ]),
      color: '#34d399',
    })

    // 2 Crimson streams from right
    streams.push({
      curve: new THREE.CatmullRomCurve3([
        new THREE.Vector3(3.2, 1.6, 0.2),
        new THREE.Vector3(1.9, 0.8, -0.3),
        new THREE.Vector3(0.6, 0.2, 0),
        new THREE.Vector3(0, 0, 0),
      ]),
      color: '#f43f5e',
    })
    streams.push({
      curve: new THREE.CatmullRomCurve3([
        new THREE.Vector3(2.9, -1.8, -0.4),
        new THREE.Vector3(1.6, -0.9, 0.3),
        new THREE.Vector3(0.5, -0.2, 0),
        new THREE.Vector3(0, 0, 0),
      ]),
      color: '#fb7185',
    })

    return streams
  }, [])

  const packetGroupRef = useRef<THREE.Group>(null)

  useFrame(({ clock }) => {
    if (reducedMotion || document.hidden || !packetGroupRef.current) return
    const t = clock.getElapsedTime()

    packetGroupRef.current.children.forEach((child, idx) => {
      const stream = streamPoints[idx % streamPoints.length]
      const progress = (t * 0.4 + idx * 0.25) % 1
      const pos = stream.curve.getPointAt(progress)
      child.position.copy(pos)
      const scale = Math.sin(progress * Math.PI) * 0.08 + 0.02
      child.scale.set(scale, scale, scale)
    })
  })

  return (
    <group>
      {/* Stream Arc Ribbons */}
      {streamPoints.map((stream, idx) => {
        const points = stream.curve.getPoints(36)
        const geometry = new THREE.BufferGeometry().setFromPoints(points)
        return (
          <primitive
            key={idx}
            object={
              new THREE.Line(
                geometry,
                new THREE.LineBasicMaterial({
                  color: stream.color,
                  transparent: true,
                  opacity: 0.25,
                  linewidth: 1,
                })
              )
            }
          />
        )
      })}

      {/* Traveling Data Packets */}
      <group ref={packetGroupRef}>
        {streamPoints.flatMap((stream, sIdx) =>
          [0, 1].map((pIdx) => (
            <mesh key={`${sIdx}-${pIdx}`}>
              <sphereGeometry args={[1, 12, 12]} />
              <meshBasicMaterial color={stream.color} />
            </mesh>
          ))
        )}
      </group>
    </group>
  )
}

// ----------------------------------------------------------------------
// 5. Ambient Deep Space Dust Particles
// ----------------------------------------------------------------------

function AmbientParticles({ reducedMotion }: { reducedMotion: boolean }) {
  const pointsRef = useRef<THREE.Points>(null)

  useFrame(({ clock }) => {
    if (reducedMotion || document.hidden || !pointsRef.current) return
    pointsRef.current.rotation.y = clock.getElapsedTime() * 0.02
  })

  return (
    <points ref={pointsRef}>
      <bufferGeometry>
        <bufferAttribute attach="attributes-position" args={[STATIC_DUST_POSITIONS, 3]} />
      </bufferGeometry>
      <pointsMaterial size={0.035} color="#94a3b8" transparent opacity={0.45} sizeAttenuation />
    </points>
  )
}

// ----------------------------------------------------------------------
// 6. Interactive Parallax Pivot Container
// ----------------------------------------------------------------------

function ParallaxPivot({ children, reducedMotion }: { children: ReactNode; reducedMotion: boolean }) {
  const pivotRef = useRef<THREE.Group>(null)
  const { pointer } = useThree()

  useFrame(() => {
    if (reducedMotion || document.hidden || !pivotRef.current) return
    const targetY = pointer.x * 0.25
    const targetX = -pointer.y * 0.18
    pivotRef.current.rotation.y = THREE.MathUtils.lerp(pivotRef.current.rotation.y, targetY, 0.05)
    pivotRef.current.rotation.x = THREE.MathUtils.lerp(pivotRef.current.rotation.x, targetX, 0.05)
  })

  return <group ref={pivotRef}>{children}</group>
}

// ----------------------------------------------------------------------
// Main 3D Canvas Scene Export
// ----------------------------------------------------------------------

export function Insight3DScene() {
  const [hasWebGL] = useState<boolean>(() => isWebGLAvailable())
  const [reducedMotion, setReducedMotion] = useState<boolean>(() => {
    if (typeof window === 'undefined') return false
    return window.matchMedia('(prefers-reduced-motion: reduce)').matches
  })

  useEffect(() => {
    const mq = window.matchMedia('(prefers-reduced-motion: reduce)')
    const handler = (e: MediaQueryListEvent) => setReducedMotion(e.matches)
    mq.addEventListener('change', handler)
    return () => mq.removeEventListener('change', handler)
  }, [])

  if (!hasWebGL) {
    return <SceneFallback />
  }

  return (
    <WebGLErrorBoundary fallback={<SceneFallback />}>
      <div className="relative w-full h-full min-h-[460px] overflow-hidden bg-slate-950 select-none">
        {/* Cinematic Backdrop Glow Gradients */}
        <div className="absolute inset-0 pointer-events-none bg-radial from-slate-900 via-slate-950 to-slate-950 opacity-90" />
        <div className="absolute top-1/4 -left-20 size-80 rounded-full bg-emerald-500/10 blur-3xl pointer-events-none" />
        <div className="absolute bottom-1/4 -right-20 size-80 rounded-full bg-rose-500/10 blur-3xl pointer-events-none" />
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 size-96 rounded-full bg-indigo-500/10 blur-3xl pointer-events-none" />

        {/* Interactive 3D Canvas */}
        <Canvas
          camera={{ position: [0, 0, 7.5], fov: 46 }}
          dpr={[1, 1.5]}
          gl={{ antialias: true, alpha: true, powerPreference: 'high-performance' }}
          className="relative z-10 w-full h-full cursor-grab active:cursor-grabbing"
          style={{ pointerEvents: 'auto' }}
        >
          {/* Balanced Restrained Lighting */}
          <ambientLight intensity={0.4} />
          <directionalLight position={[5, 6, 4]} intensity={0.8} />
          <directionalLight position={[-5, -4, -2]} intensity={0.4} color="#6366f1" />

          {/* Interactive Parallax Scene Pivot */}
          <ParallaxPivot reducedMotion={reducedMotion}>
            {/* Core Elements */}
            <InsightCore reducedMotion={reducedMotion} />
            <DataStreams reducedMotion={reducedMotion} />
            <AmbientParticles reducedMotion={reducedMotion} />

            {/* Positive Review Elements (The Good) */}
            <PositiveReviewBubble
              initialPos={[-2.8, 1.7, 0]}
              text="Excellent build quality"
              stars="★ 5.0"
              aspect="Hardware"
              floatSpeed={1.0}
              phaseOffset={0}
              reducedMotion={reducedMotion}
            />
            <PositiveReviewBubble
              initialPos={[-2.4, -1.5, 0.4]}
              text="Battery lasts 3 days"
              stars="★ 5.0"
              aspect="Power"
              floatSpeed={0.85}
              phaseOffset={1.8}
              reducedMotion={reducedMotion}
            />
            <PositiveReviewBubble
              initialPos={[-3.3, 0.1, -0.3]}
              text="Super intuitive UX"
              stars="★ 5.0"
              aspect="App"
              floatSpeed={1.1}
              phaseOffset={3.2}
              reducedMotion={reducedMotion}
            />

            {/* Negative Review Elements (The Bad) */}
            <NegativeReviewCard
              initialPos={[2.7, 1.6, 0.2]}
              text="Battery drains in 2h"
              stars="★ 1.0"
              aspect="Power"
              floatSpeed={0.9}
              phaseOffset={0.7}
              reducedMotion={reducedMotion}
            />
            <NegativeReviewCard
              initialPos={[2.5, -1.6, -0.2]}
              text="Bluetooth drops"
              stars="★ 2.0"
              aspect="Connectivity"
              floatSpeed={1.05}
              phaseOffset={2.4}
              reducedMotion={reducedMotion}
            />
            <NegativeReviewCard
              initialPos={[3.2, 0.1, 0.3]}
              text="Hinge cracked on day 4"
              stars="★ 1.0"
              aspect="Mechanical"
              floatSpeed={0.95}
              phaseOffset={4.0}
              reducedMotion={reducedMotion}
            />
          </ParallaxPivot>
        </Canvas>

        {/* Ambient Overlay Taglines */}
        <div className="absolute top-6 left-6 z-20 pointer-events-none">
          <div className="inline-flex items-center gap-2 rounded-full border border-indigo-500/30 bg-slate-900/80 px-3 py-1 text-[11px] font-semibold text-indigo-300 shadow-md backdrop-blur-md">
            <span className="size-2 rounded-full bg-emerald-400 animate-pulse" />
            <span>The Good. The Bad. The Insight.</span>
          </div>
        </div>

        <div className="absolute bottom-6 left-6 right-6 z-20 pointer-events-none flex items-center justify-between text-[11px] text-slate-400">
          <span className="backdrop-blur-sm bg-slate-950/60 px-2 py-0.5 rounded border border-slate-800/40">
            Real-time R3F Neural Engine · Drag or move pointer to orbit
          </span>
          <span className="hidden sm:inline backdrop-blur-sm bg-slate-950/60 px-2 py-0.5 rounded border border-slate-800/40 text-indigo-400">
            Powered by InsightForge NLP
          </span>
        </div>
      </div>
    </WebGLErrorBoundary>
  )
}
