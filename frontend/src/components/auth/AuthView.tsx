import React, { useState } from 'react'
import {
  Lock,
  Mail,
  ArrowRight,
  ShieldCheck,
  CheckCircle2,
  AlertCircle,
  Eye,
  EyeOff,
} from 'lucide-react'
import { motion, AnimatePresence } from 'motion/react'
import { useAuth } from '../../context/useAuth'
import { CursorNetworkBackground } from './CursorNetworkBackground'

type AuthMode = 'login' | 'signup' | 'forgot_password' | 'update_password'

interface AuthViewProps {
  dark?: boolean
}

export function AuthView({ dark = true }: AuthViewProps) {
  const {
    signIn,
    signUp,
    resetPasswordForEmail,
    updatePassword,
    isSupabaseConfigured,
    isPasswordRecovery,
    setIsPasswordRecovery,
  } = useAuth()

  const [userSelectedMode, setUserSelectedMode] = useState<AuthMode | null>(null)
  const mode: AuthMode = userSelectedMode ?? (isPasswordRecovery ? 'update_password' : 'login')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [confirmPassword, setConfirmPassword] = useState('')
  const [showPassword, setShowPassword] = useState(false)

  const [submitting, setSubmitting] = useState(false)
  const [errorMsg, setErrorMsg] = useState<string | null>(null)
  const [successMsg, setSuccessMsg] = useState<string | null>(null)
  const [verificationPending, setVerificationPending] = useState(false)

  function resetForm() {
    setErrorMsg(null)
    setSuccessMsg(null)
    setVerificationPending(false)
  }

  function switchMode(newMode: AuthMode) {
    resetForm()
    setUserSelectedMode(newMode)
    if (newMode !== 'update_password') {
      setIsPasswordRecovery(false)
    }
  }

  function validateEmail(val: string): boolean {
    return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(val.trim())
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    setErrorMsg(null)
    setSuccessMsg(null)

    // Form validations
    if (mode !== 'update_password') {
      if (!email.trim()) {
        setErrorMsg('Please enter your email address.')
        return
      }
      if (!validateEmail(email)) {
        setErrorMsg('Please provide a valid email format (e.g., name@company.com).')
        return
      }
    }

    if (mode === 'login') {
      if (!password) {
        setErrorMsg('Please enter your password.')
        return
      }
      setSubmitting(true)
      const { error } = await signIn(email, password)
      setSubmitting(false)
      if (error) {
        // Safe, non-sensitive error message
        setErrorMsg('Invalid email or password credentials. Please verify and try again.')
      }
    } else if (mode === 'signup') {
      if (!password || password.length < 6) {
        setErrorMsg('Password must be at least 6 characters long.')
        return
      }
      if (password !== confirmPassword) {
        setErrorMsg('Passwords do not match. Please re-enter.')
        return
      }
      setSubmitting(true)
      const { error, needsEmailVerification } = await signUp(email, password)
      setSubmitting(false)
      if (error) {
        setErrorMsg(error.message || 'Unable to complete registration. Please try again.')
      } else if (needsEmailVerification) {
        setVerificationPending(true)
        setSuccessMsg(
          'Account created successfully! We sent a confirmation link to your email. Please verify to sign in.'
        )
      }
    } else if (mode === 'forgot_password') {
      setSubmitting(true)
      const { error } = await resetPasswordForEmail(email)
      setSubmitting(false)
      if (error) {
        // Non-revealing error message to avoid email enumeration
        setSuccessMsg(
          'If an account exists with this email, a password reset link has been dispatched to your inbox.'
        )
      } else {
        setSuccessMsg(
          'If an account exists with this email, a password reset link has been dispatched to your inbox.'
        )
      }
    } else if (mode === 'update_password') {
      if (!password || password.length < 6) {
        setErrorMsg('New password must be at least 6 characters long.')
        return
      }
      if (password !== confirmPassword) {
        setErrorMsg('Passwords do not match.')
        return
      }
      setSubmitting(true)
      const { error } = await updatePassword(password)
      setSubmitting(false)
      if (error) {
        setErrorMsg(error.message || 'Failed to update password. Recovery link may have expired.')
      } else {
        setSuccessMsg('Your password has been successfully updated. You may now continue.')
        setTimeout(() => {
          switchMode('login')
        }, 2000)
      }
    }
  }

  const surface = dark
    ? 'bg-slate-900/80 border-slate-800 text-slate-100 shadow-2xl backdrop-blur-xl'
    : 'bg-white/90 border-slate-200 text-slate-900 shadow-2xl backdrop-blur-xl'
  const muted = dark ? 'text-slate-400' : 'text-slate-500'
  const inputBg = dark
    ? 'bg-slate-950/80 border-slate-800 text-slate-100 placeholder-slate-500 focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500'
    : 'bg-slate-50/90 border-slate-200 text-slate-900 placeholder-slate-400 focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500'

  return (
    <div className="relative min-h-screen w-full flex flex-col items-center justify-center bg-slate-950 text-slate-100 px-4 py-10 sm:px-6 sm:py-12">
      {/* Always-on cursor-reactive network animation covering the whole page behind the card */}
      <CursorNetworkBackground />

      <div className="relative z-10 w-full max-w-md flex flex-col items-center">
        {/* InsightForge Logo Block - Centered above the card */}
        <div className="mb-6 sm:mb-8 flex flex-col items-center text-center">
          <div className="flex size-12 items-center justify-center rounded-2xl bg-gradient-to-tr from-indigo-600 via-indigo-500 to-cyan-400 text-2xl font-black text-white shadow-xl shadow-indigo-500/35 border border-white/10 mb-3">
            I
          </div>
          <div className="flex items-center justify-center gap-2 text-2xl font-black tracking-tight text-white">
            <span>InsightForge</span>
            <span className="rounded-md bg-indigo-500/20 border border-indigo-500/40 px-1.5 py-0.5 text-[10px] font-bold text-indigo-300 tracking-normal">
              v2.0
            </span>
          </div>
          <p className="mt-1 text-xs text-slate-400 font-medium">
            AI Review Intelligence & Defect Analytics
          </p>
        </div>

        {/* Configuration Notice if Supabase keys not set */}
        {!isSupabaseConfigured && (
          <div className="w-full mb-6 rounded-xl border border-amber-500/30 bg-amber-500/10 p-4 text-xs text-amber-300">
            <div className="flex items-center gap-2 font-bold mb-1">
              <AlertCircle size={15} /> Supabase Configuration Required
            </div>
            <p className="leading-relaxed">
              Please set <code className="bg-amber-500/20 px-1 py-0.5 rounded font-mono">VITE_SUPABASE_URL</code> and{' '}
              <code className="bg-amber-500/20 px-1 py-0.5 rounded font-mono">VITE_SUPABASE_ANON_KEY</code> in{' '}
              <code className="bg-amber-500/20 px-1 py-0.5 rounded font-mono">frontend/.env.local</code> to enable
              production authentication.
            </p>
          </div>
        )}

        {/* Main Auth Card with Motion Transitions */}
        <div className={`w-full rounded-3xl border p-8 sm:p-10 ${surface}`}>
            <AnimatePresence mode="wait">
              <motion.div
                key={mode}
                initial={{ opacity: 0, y: 12 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -12 }}
                transition={{ duration: 0.2 }}
              >
                {/* Mode Title & Subtitle */}
                <div className="mb-6">
                  <h1 className="text-xl sm:text-2xl font-black tracking-tight text-white">
                    {mode === 'login' && 'Sign In to Workspace'}
                    {mode === 'signup' && 'Create Your Account'}
                    {mode === 'forgot_password' && 'Reset Password'}
                    {mode === 'update_password' && 'Set New Password'}
                  </h1>
                  <p className={`text-xs mt-1.5 ${muted} leading-relaxed`}>
                    {mode === 'login' && 'Enter your credentials to access your private analytics workspace.'}
                    {mode === 'signup' && 'Create a private, isolated workspace for your review telemetry and defects.'}
                    {mode === 'forgot_password' &&
                      'Enter your registered email and we will dispatch a secure recovery link.'}
                    {mode === 'update_password' && 'Enter your new password to regain access to your workspace.'}
                  </p>
                </div>

                {/* Feedback Alerts */}
                {errorMsg && (
                  <div className="mb-5 flex items-start gap-2.5 rounded-xl border border-rose-500/30 bg-rose-500/10 p-3.5 text-xs text-rose-300 animate-fadeIn">
                    <AlertCircle size={16} className="shrink-0 mt-0.5 text-rose-400" />
                    <span>{errorMsg}</span>
                  </div>
                )}

                {successMsg && (
                  <div className="mb-5 flex items-start gap-2.5 rounded-xl border border-emerald-500/30 bg-emerald-500/10 p-3.5 text-xs text-emerald-300 animate-fadeIn">
                    <CheckCircle2 size={16} className="shrink-0 mt-0.5 text-emerald-400" />
                    <span>{successMsg}</span>
                  </div>
                )}

                {/* Verification Notice Flow */}
                {verificationPending ? (
                  <div className="text-center py-6 space-y-4">
                    <div className="size-14 rounded-2xl bg-indigo-500/15 text-indigo-400 mx-auto flex items-center justify-center border border-indigo-500/30">
                      <Mail size={28} />
                    </div>
                    <div className="space-y-1">
                      <h3 className="font-bold text-sm text-white">Check Your Email</h3>
                      <p className={`text-xs ${muted}`}>
                        We sent an activation link to your inbox. Please click it to confirm your account.
                      </p>
                    </div>
                    <button
                      type="button"
                      onClick={() => switchMode('login')}
                      className="w-full rounded-xl bg-indigo-600 px-4 py-2.5 text-xs font-bold text-white hover:bg-indigo-500 transition shadow-lg shadow-indigo-600/30"
                    >
                      Back to Sign In
                    </button>
                  </div>
                ) : (
                  <form onSubmit={handleSubmit} className="space-y-4" noValidate>
                    {/* Email Input */}
                    {mode !== 'update_password' && (
                      <div>
                        <label htmlFor="auth-email" className={`block text-xs font-semibold mb-1.5 ${muted}`}>
                          Work Email Address
                        </label>
                        <div className="relative">
                          <Mail size={16} className="absolute left-3.5 top-3.5 text-slate-500" />
                          <input
                            id="auth-email"
                            type="email"
                            autoComplete="email"
                            value={email}
                            onChange={(e) => setEmail(e.target.value)}
                            placeholder="alex@enterprise.com"
                            disabled={submitting}
                            className={`w-full rounded-xl border pl-10 pr-3.5 py-3 text-xs focus:outline-none transition ${inputBg}`}
                          />
                        </div>
                      </div>
                    )}

                    {/* Password Input */}
                    {mode !== 'forgot_password' && (
                      <div>
                        <div className="flex items-center justify-between mb-1.5">
                          <label htmlFor="auth-password" className={`text-xs font-semibold ${muted}`}>
                            {mode === 'update_password' ? 'New Password' : 'Password'}
                          </label>
                          {mode === 'login' && (
                            <button
                              type="button"
                              onClick={() => switchMode('forgot_password')}
                              className="text-[11px] font-medium text-indigo-400 hover:text-indigo-300 transition"
                            >
                              Forgot password?
                            </button>
                          )}
                        </div>
                        <div className="relative">
                          <Lock size={16} className="absolute left-3.5 top-3.5 text-slate-500" />
                          <input
                            id="auth-password"
                            type={showPassword ? 'text' : 'password'}
                            autoComplete={mode === 'signup' ? 'new-password' : 'current-password'}
                            value={password}
                            onChange={(e) => setPassword(e.target.value)}
                            placeholder="••••••••••••"
                            disabled={submitting}
                            className={`w-full rounded-xl border pl-10 pr-11 py-3 text-xs focus:outline-none transition ${inputBg}`}
                          />
                          <button
                            type="button"
                            onClick={() => setShowPassword(!showPassword)}
                            className="absolute right-3.5 top-3.5 text-slate-500 hover:text-slate-300"
                            aria-label={showPassword ? 'Hide password' : 'Show password'}
                          >
                            {showPassword ? <EyeOff size={16} /> : <Eye size={16} />}
                          </button>
                        </div>
                      </div>
                    )}

                    {/* Confirm Password Input */}
                    {(mode === 'signup' || mode === 'update_password') && (
                      <div>
                        <label htmlFor="auth-confirm-password" className={`block text-xs font-semibold mb-1.5 ${muted}`}>
                          Confirm Password
                        </label>
                        <div className="relative">
                          <Lock size={16} className="absolute left-3.5 top-3.5 text-slate-500" />
                          <input
                            id="auth-confirm-password"
                            type={showPassword ? 'text' : 'password'}
                            autoComplete="new-password"
                            value={confirmPassword}
                            onChange={(e) => setConfirmPassword(e.target.value)}
                            placeholder="••••••••••••"
                            disabled={submitting}
                            className={`w-full rounded-xl border pl-10 pr-3.5 py-3 text-xs focus:outline-none transition ${inputBg}`}
                          />
                        </div>
                      </div>
                    )}

                    {/* Action Submit Button */}
                    <button
                      type="submit"
                      disabled={submitting || !isSupabaseConfigured}
                      className="w-full mt-2 inline-flex items-center justify-center gap-2 rounded-xl bg-gradient-to-r from-indigo-600 to-indigo-500 px-4 py-3 text-xs font-bold text-white shadow-lg shadow-indigo-600/30 hover:from-indigo-500 hover:to-indigo-400 disabled:opacity-50 disabled:cursor-not-allowed transition transform active:scale-[0.99]"
                    >
                      {submitting ? (
                        <span className="flex items-center gap-2">
                          <span className="size-3.5 rounded-full border-2 border-white/30 border-t-white animate-spin" />
                          Authenticating...
                        </span>
                      ) : (
                        <>
                          {mode === 'login' && 'Sign In to Workspace'}
                          {mode === 'signup' && 'Create Free Account'}
                          {mode === 'forgot_password' && 'Send Recovery Instructions'}
                          {mode === 'update_password' && 'Save New Password'}
                          <ArrowRight size={14} />
                        </>
                      )}
                    </button>
                  </form>
                )}

                {/* Footer Switcher */}
                <div className="mt-6 border-t border-slate-800/80 pt-4 text-center text-xs">
                  {mode === 'login' && (
                    <p className={muted}>
                      Don't have an account yet?{' '}
                      <button
                        type="button"
                        onClick={() => switchMode('signup')}
                        className="font-bold text-indigo-400 hover:text-indigo-300 transition"
                      >
                        Create one now
                      </button>
                    </p>
                  )}

                  {mode === 'signup' && (
                    <p className={muted}>
                      Already have an account?{' '}
                      <button
                        type="button"
                        onClick={() => switchMode('login')}
                        className="font-bold text-indigo-400 hover:text-indigo-300 transition"
                      >
                        Sign in
                      </button>
                    </p>
                  )}

                  {(mode === 'forgot_password' || mode === 'update_password') && (
                    <p className={muted}>
                      Remember your password?{' '}
                      <button
                        type="button"
                        onClick={() => switchMode('login')}
                        className="font-bold text-indigo-400 hover:text-indigo-300 transition"
                      >
                        Back to sign in
                      </button>
                    </p>
                  )}
                </div>
              </motion.div>
            </AnimatePresence>
          </div>

          {/* Security & RLS Assurance Footer */}
          <div className={`mt-6 text-center text-[11px] ${muted} flex items-center justify-center gap-2`}>
            <ShieldCheck size={14} className="text-emerald-500" />
            <span>Protected by Supabase Auth with Row Level Security (RLS)</span>
          </div>
        </div>
      </div>
    )
  }
