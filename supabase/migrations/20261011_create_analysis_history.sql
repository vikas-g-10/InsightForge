-- ==============================================================================
-- InsightForge: Secure User Analysis History with Row Level Security (RLS)
-- Migration: 20261011_create_analysis_history.sql
-- ==============================================================================

-- 1. Create analysis_history table tied directly to auth.users
CREATE TABLE IF NOT EXISTS public.analysis_history (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    source_name TEXT NOT NULL,
    total_reviews INTEGER NOT NULL DEFAULT 0,
    avg_rating NUMERIC(3, 2),
    safety_alerts INTEGER NOT NULL DEFAULT 0,
    result JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- 2. Performance indexes for user-isolated queries ordered by creation time
CREATE INDEX IF NOT EXISTS idx_analysis_history_user_created 
    ON public.analysis_history(user_id, created_at DESC);

-- 3. Enable Row Level Security (RLS)
ALTER TABLE public.analysis_history ENABLE ROW LEVEL SECURITY;

-- 4. RLS Policies: Strictly enforce auth.uid() = user_id for all operations

-- SELECT Policy: Users can only view their own analysis history
DROP POLICY IF EXISTS "Users can view own analysis history" ON public.analysis_history;
CREATE POLICY "Users can view own analysis history"
    ON public.analysis_history
    FOR SELECT
    TO authenticated
    USING (auth.uid() = user_id);

-- INSERT Policy: Users can only insert records where user_id matches their authenticated UID
DROP POLICY IF EXISTS "Users can insert own analysis history" ON public.analysis_history;
CREATE POLICY "Users can insert own analysis history"
    ON public.analysis_history
    FOR INSERT
    TO authenticated
    WITH CHECK (auth.uid() = user_id);

-- UPDATE Policy: Users can only update their own records
DROP POLICY IF EXISTS "Users can update own analysis history" ON public.analysis_history;
CREATE POLICY "Users can update own analysis history"
    ON public.analysis_history
    FOR UPDATE
    TO authenticated
    USING (auth.uid() = user_id)
    WITH CHECK (auth.uid() = user_id);

-- DELETE Policy: Users can only delete their own records
DROP POLICY IF EXISTS "Users can delete own analysis history" ON public.analysis_history;
CREATE POLICY "Users can delete own analysis history"
    ON public.analysis_history
    FOR DELETE
    TO authenticated
    USING (auth.uid() = user_id);

-- 5. Revoke public/anon access completely
REVOKE ALL ON public.analysis_history FROM anon;
GRANT SELECT, INSERT, UPDATE, DELETE ON public.analysis_history TO authenticated;
