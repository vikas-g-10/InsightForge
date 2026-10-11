import { supabase, isSupabaseConfigured } from '../supabase'
import type { AnalyzeResponse } from '../api'

export interface HistoryItem {
  id: string
  timestamp: string
  sourceName: string
  totalReviews: number
  avgRating: number
  safetyAlerts: number
  result: AnalyzeResponse
}

function getUserCacheKey(userId: string): string {
  return `insightforge-history-${userId}`
}

export async function fetchUserHistory(userId: string): Promise<HistoryItem[]> {
  if (!userId) return []

  // 1. If Supabase is configured, fetch from public.analysis_history (RLS enforced)
  if (isSupabaseConfigured) {
    try {
      const { data, error } = await supabase
        .from('analysis_history')
        .select('*')
        .order('created_at', { ascending: false })
        .limit(20)

      if (!error && Array.isArray(data)) {
        const mapped: HistoryItem[] = data.map((row) => ({
          id: String(row.id),
          timestamp: new Date(row.created_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
          sourceName: row.source_name,
          totalReviews: row.total_reviews,
          avgRating: Number(row.avg_rating ?? 0),
          safetyAlerts: row.safety_alerts,
          result: row.result as AnalyzeResponse,
        }))

        // Cache locally under user-scoped key
        try {
          localStorage.setItem(getUserCacheKey(userId), JSON.stringify(mapped))
        } catch {
          // ignore storage quota issues
        }

        return mapped
      }
    } catch (err) {
      console.warn('Failed to fetch history from Supabase, falling back to local user cache:', err)
    }
  }

  // 2. Fallback to user-isolated localStorage key
  try {
    const raw = localStorage.getItem(getUserCacheKey(userId))
    return raw ? JSON.parse(raw) : []
  } catch {
    return []
  }
}

export async function saveUserHistory(
  userId: string,
  item: {
    sourceName: string
    totalReviews: number
    avgRating: number
    safetyAlerts: number
    result: AnalyzeResponse
  }
): Promise<HistoryItem> {
  const localId = `run-${Date.now()}`
  const now = new Date()
  const localItem: HistoryItem = {
    id: localId,
    timestamp: now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    sourceName: item.sourceName,
    totalReviews: item.totalReviews,
    avgRating: item.avgRating,
    safetyAlerts: item.safetyAlerts,
    result: item.result,
  }

  if (isSupabaseConfigured && userId) {
    try {
      const { data, error } = await supabase
        .from('analysis_history')
        .insert({
          user_id: userId,
          source_name: item.sourceName,
          total_reviews: item.totalReviews,
          avg_rating: item.avgRating,
          safety_alerts: item.safetyAlerts,
          result: item.result,
        })
        .select()
        .single()

      if (!error && data) {
        localItem.id = String(data.id)
      }
    } catch (err) {
      console.warn('Failed to insert history to Supabase, saving to user local cache:', err)
    }
  }

  // Save in user-isolated localStorage
  try {
    const cacheKey = getUserCacheKey(userId)
    const existing: HistoryItem[] = JSON.parse(localStorage.getItem(cacheKey) || '[]')
    const updated = [localItem, ...existing.slice(0, 19)]
    localStorage.setItem(cacheKey, JSON.stringify(updated))
  } catch {
    // ignore
  }

  return localItem
}

export function clearUserHistoryCache(userId: string): void {
  if (userId) {
    try {
      localStorage.removeItem(getUserCacheKey(userId))
    } catch {
      // ignore
    }
  }
}
