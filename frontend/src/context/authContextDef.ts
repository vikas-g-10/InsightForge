import { createContext } from 'react'
import type { User, Session } from '@supabase/supabase-js'

export interface AuthContextType {
  user: User | null
  session: Session | null
  accessToken: string | null
  loading: boolean
  isSupabaseConfigured: boolean
  isPasswordRecovery: boolean
  setIsPasswordRecovery: (value: boolean) => void
  signUp: (email: string, password: string) => Promise<{ error: Error | null; needsEmailVerification?: boolean }>
  signIn: (email: string, password: string) => Promise<{ error: Error | null }>
  signOut: () => Promise<void>
  resetPasswordForEmail: (email: string) => Promise<{ error: Error | null }>
  updatePassword: (newPassword: string) => Promise<{ error: Error | null }>
  getAccessToken: () => Promise<string | null>
}

export const AuthContext = createContext<AuthContextType | undefined>(undefined)
