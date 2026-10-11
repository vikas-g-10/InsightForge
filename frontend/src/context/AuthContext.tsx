import React, { useEffect, useState, useCallback } from 'react'
import type { User, Session, AuthChangeEvent } from '@supabase/supabase-js'
import { supabase, isSupabaseConfigured } from '../supabase'
import { AuthContext, type AuthContextType } from './authContextDef'

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<User | null>(null)
  const [session, setSession] = useState<Session | null>(null)
  const [loading, setLoading] = useState<boolean>(() => isSupabaseConfigured)
  const [isPasswordRecovery, setIsPasswordRecovery] = useState<boolean>(false)

  // Initialize session and listen for auth state changes
  useEffect(() => {
    let mounted = true

    if (!isSupabaseConfigured) {
      return
    }

    // 1. Check initial active session
    supabase.auth.getSession().then(({ data: { session: initialSession }, error }) => {
      if (!mounted) return
      if (error) {
        console.warn('Failed to retrieve initial Supabase session:', error.message)
      }
      setSession(initialSession)
      setUser(initialSession?.user ?? null)
      setLoading(false)
    }).catch((err) => {
      console.warn('Unexpected error in getSession:', err)
      if (mounted) setLoading(false)
    })

    // 2. Listen to state changes (SIGN_IN, SIGN_OUT, TOKEN_REFRESHED, PASSWORD_RECOVERY, etc.)
    const {
      data: { subscription },
    } = supabase.auth.onAuthStateChange((event: AuthChangeEvent, currentSession: Session | null) => {
      if (!mounted) return
      setSession(currentSession)
      setUser(currentSession?.user ?? null)
      setLoading(false)

      if (event === 'PASSWORD_RECOVERY') {
        setIsPasswordRecovery(true)
      } else if (event === 'SIGNED_OUT') {
        setIsPasswordRecovery(false)
      }
    })

    return () => {
      mounted = false
      subscription.unsubscribe()
    }
  }, [])

  const getAccessToken = useCallback(async (): Promise<string | null> => {
    if (!isSupabaseConfigured) return null

    // If current session is still valid (not expired within 60s), use it
    if (session?.access_token && session.expires_at) {
      const nowSec = Math.floor(Date.now() / 1000)
      if (session.expires_at - nowSec > 60) {
        return session.access_token
      }
    }

    // Otherwise refresh session to guarantee fresh token
    try {
      const { data, error } = await supabase.auth.getSession()
      if (error || !data.session) {
        return null
      }
      setSession(data.session)
      setUser(data.session.user)
      return data.session.access_token
    } catch {
      return null
    }
  }, [session])

  const signUp = async (email: string, password: string) => {
    if (!isSupabaseConfigured) {
      return { error: new Error('Supabase is not configured. Please set VITE_SUPABASE_URL and VITE_SUPABASE_ANON_KEY.') }
    }
    try {
      const { data, error } = await supabase.auth.signUp({
        email: email.trim(),
        password,
      })
      if (error) return { error: new Error(error.message) }

      // If Supabase has email confirmation enabled, user is created but session is null
      const needsEmailVerification = Boolean(data.user && !data.session)
      return { error: null, needsEmailVerification }
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Sign-up failed'
      return { error: new Error(message) }
    }
  }

  const signIn = async (email: string, password: string) => {
    if (!isSupabaseConfigured) {
      return { error: new Error('Supabase is not configured. Please set VITE_SUPABASE_URL and VITE_SUPABASE_ANON_KEY.') }
    }
    try {
      const { data, error } = await supabase.auth.signInWithPassword({
        email: email.trim(),
        password,
      })
      if (error) return { error: new Error(error.message) }
      setSession(data.session)
      setUser(data.user)
      return { error: null }
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Sign-in failed'
      return { error: new Error(message) }
    }
  }

  const signOut = async () => {
    if (!isSupabaseConfigured) {
      setUser(null)
      setSession(null)
      return
    }
    try {
      await supabase.auth.signOut()
    } catch (err) {
      console.warn('Error during Supabase sign-out:', err)
    } finally {
      setUser(null)
      setSession(null)
      setIsPasswordRecovery(false)
    }
  }

  const resetPasswordForEmail = async (email: string) => {
    if (!isSupabaseConfigured) {
      return { error: new Error('Supabase is not configured.') }
    }
    try {
      const { error } = await supabase.auth.resetPasswordForEmail(email.trim(), {
        redirectTo: `${window.location.origin}/`,
      })
      if (error) return { error: new Error(error.message) }
      return { error: null }
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Password reset request failed'
      return { error: new Error(message) }
    }
  }

  const updatePassword = async (newPassword: string) => {
    if (!isSupabaseConfigured) {
      return { error: new Error('Supabase is not configured.') }
    }
    try {
      const { error } = await supabase.auth.updateUser({ password: newPassword })
      if (error) return { error: new Error(error.message) }
      setIsPasswordRecovery(false)
      return { error: null }
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Password update failed'
      return { error: new Error(message) }
    }
  }

  const value: AuthContextType = {
    user,
    session,
    accessToken: session?.access_token ?? null,
    loading,
    isSupabaseConfigured,
    isPasswordRecovery,
    setIsPasswordRecovery,
    signUp,
    signIn,
    signOut,
    resetPasswordForEmail,
    updatePassword,
    getAccessToken,
  }

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>
}
