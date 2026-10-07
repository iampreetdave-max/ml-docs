-- Run once in Supabase Dashboard → SQL Editor.

-- 1. Table for review text (Postgres `text` has no length limit)
create table if not exists public.reviews (
  id          uuid primary key default gen_random_uuid(),
  created_at  timestamptz not null default now(),
  message     text not null default '',
  page        text,
  files       jsonb not null default '[]'::jsonb   -- storage paths of uploaded files
);

alter table public.reviews enable row level security;

-- Visitors may only INSERT. They can't read, edit or delete anyone's reviews.
-- You read them in the dashboard (service role bypasses RLS).
create policy "anon can submit reviews"
  on public.reviews for insert
  to anon
  with check (true);

-- 2. Private bucket for attachments (any file type)
insert into storage.buckets (id, name, public)
values ('review-files', 'review-files', false)
on conflict (id) do nothing;

-- Visitors may only upload into this bucket. No listing/downloading.
create policy "anon can upload review files"
  on storage.objects for insert
  to anon
  with check (bucket_id = 'review-files');
