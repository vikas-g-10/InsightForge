# PRD v4: Revora AI --- Product Review Insight Analyzer

Oct 10, 2026 · Project owner: Vikas · Team SPARTANS · Problem Statement
7 · AI/ML

**Architecture revision:** Google Stitch for UI design, React +
TypeScript + Tailwind CSS frontend, Python + FastAPI backend, and
Supabase PostgreSQL for persistent analysis history and metadata. The
10-hour target and free-tier constraint remain. This PRD supersedes
earlier versions where architecture decisions conflict.

## What Changed in v4

I re-read v2 looking for anything that could stop a working prototype at
hour 10. Fixes:

-   **Frontend/backend separated:** React + TypeScript + Tailwind CSS is
    the user interface; FastAPI owns validation, analysis APIs,
    simulator calculations, and exports.
-   **Google Stitch is a design tool, not a runtime dependency:**
    translate the chosen screens into working React components.
-   **Supabase PostgreSQL added:** persist analysis records and report
    metadata so users can revisit past analyses. Uploaded review files
    are not stored permanently by default.
-   **Lite-first delivery remains:** complete a simple working vertical
    slice before integrating heavier NLP models.
-   **Data contract added** (tables, columns, function signature) so
    teammates can build in parallel without waiting on each other.
-   **Demo data recipe added,** including a test that checks the
    simulator recovers effects we planted. Without ratings that behave
    sensibly, the headline feature would show nonsense.
-   **Input validation, error handling and privacy** added.
-   **Milestones with go/no-go checks and a feature freeze** at hour 8.
-   **Acceptance checklist** that defines 'fully working'.
-   **Panel Q&A** added.

## Architecture Decision Record (v4)

**Chosen stack:** Google Stitch (design only) → React + TypeScript +
Vite + Tailwind CSS (Vercel frontend) → FastAPI/Python REST API
(Python-capable host) → NLP pipeline → Supabase PostgreSQL (persistent
analysis metadata/results).

**Why a database now:** the prototype should be able to save and revisit
prior analyses. The database is not required for sentiment or clustering
calculations; it is used for persistence. Keep the initial schema
minimal and avoid permanent raw-upload storage unless evidence
drill-down across sessions is a required acceptance criterion.

**Authentication scope:** no user accounts in the initial 10-hour build.
History is a demo/single-user feature unless authentication and
row-level ownership are explicitly implemented. Do not expose database
service credentials in frontend code.

**Important deployment caveat:** Vercel is the frontend target, not
automatically the FastAPI hosting target. Choose a Python host only
after verifying current free-tier runtime and dependency limits. If time
is tight, prioritize the local end-to-end prototype, then the hosted
Lite deployment.

## 1. Summary

**Product:** A compact web app that turns a pile of customer reviews
(English, Hindi, Hinglish, and Kannada, Tamil and Telugu written in
English letters) into a ranked **'what to fix first'** list with
estimated rating impact, safety escalations, and ready-to-send tickets
and replies.

**Pitch:** *We don't just summarize reviews. We tell a business what to
fix first, what it's worth, and what's dangerous, in the languages
Indian customers actually write in.*

## 2. Goals, Non-goals, Constraints

**Goals**

-   Cover every must-have in the problem statement: sentiment, topic
    extraction, grouping, insight ranking, traceability.
-   Ship four differentiators: Fix-This-First Simulator, Safety flag,
    Hinglish support, Auto-action output.
-   A fully working prototype at hour 10: runs live on a laptop, and is
    also available as a public link.

**Non-goals**

-   No paid APIs or paid cloud services are required. Supabase free-tier
    availability/limits must be verified before deployment.
-   No user accounts in the first release unless authentication is
    explicitly implemented; the initial prototype may use a
    single-user/demo workflow.
-   A database is now in scope: Supabase PostgreSQL stores analysis
    metadata and saved results. Raw uploaded files are temporary by
    default.
-   No model training or fine-tuning (no time, no labeled data).
-   No paid APIs of any kind.

**Constraints**

-   10 hours total, feature freeze at hour 8.
-   Full mode runs locally and works offline after the first model
    download.
-   Hosted frontend runs on Vercel. FastAPI is deployed to a
    Python-capable host; do not assume Vercel's serverless runtime is
    the best place for long-running NLP tasks.
-   Hosted backend must fit a free-tier memory/CPU budget, so it runs
    Lite mode and caps upload size.
-   Supabase PostgreSQL is the persistent database; secrets are
    environment variables and must never be exposed in frontend code.

**Team assumption:** 3 people. **A** NLP pipeline, **B** analytics,
simulator and actions, **C** UI, deployment and demo data. If you are 2
people, merge B into A. If solo, follow the same order and drop the
cut-list items early.

## 3. Features and Acceptance Criteria

  -----------------------------------------------------------------------
  ID                Feature           Acceptance        Priority
                                      criteria          
  ----------------- ----------------- ----------------- -----------------
  F1                Upload and ingest CSV/XLSX upload   Must
                                      or paste. Only a  
                                      review-text       
                                      column is         
                                      required; rating, 
                                      date and version  
                                      are optional and  
                                      auto-detected.    
                                      5,000 rows load   
                                      in under 5        
                                      seconds.          

  F2                Aspect-level      Each review is    Must
                    sentiment         split into        
                                      clauses; each     
                                      clause gets an    
                                      aspect and a      
                                      sentiment. 'Great 
                                      camera, terrible  
                                      battery' gives    
                                      camera positive   
                                      and battery       
                                      negative.         

  F3                Grouping          Similar comments  Must
                                      cluster under one 
                                      label with an     
                                      example sentence. 
                                      'battery dies     
                                      fast', 'poor      
                                      battery life' and 
                                      'battery 2 ghante 
                                      mein khatam' land 
                                      in one group.     

  F4                Ranking           Insights ranked   Must
                                      by frequency and  
                                      sentiment         
                                      together (see     
                                      Fix-This-First    
                                      score).           

  F5                Traceability      Click any insight Must
                                      to see the real   
                                      reviews behind    
                                      it, with the      
                                      relevant clause   
                                      highlighted.      

  F6                Fix-This-First    Per-issue         Differentiator
                    Simulator         estimated rating  
                                      uplift, plus a    
                                      slider ('fix 60%  
                                      of battery        
                                      complaints') that 
                                      updates the       
                                      projected average 
                                      rating live.      

  F7                Safety flag       Reviews           Differentiator
                                      mentioning fire,  
                                      burns, shocks,    
                                      allergic          
                                      reactions, injury 
                                      and similar are   
                                      pinned to the top 
                                      as a red alert,   
                                      regardless of     
                                      frequency.        

  F8                Hinglish support  English, Hindi,   Differentiator
                                      Hinglish plus     
                                      Kannada, Tamil    
                                      and Telugu in     
                                      English letters   
                                      (through language 
                                      packs, see 7.3a)  
                                      are analysed;     
                                      each review gets  
                                      a language badge. 

  F9                Auto-action       One click creates Differentiator
                    output            a ticket (team,   
                                      priority,         
                                      evidence, impact) 
                                      and a reply draft 
                                      for unhappy       
                                      customers; export 
                                      as Markdown/CSV.  

  F10               Sarcasm and       Suspected sarcasm Must (edge cases)
                    ambiguity         and rating/text   
                    handling          conflicts are     
                                      flagged with a    
                                      warning badge     
                                      instead of        
                                      silently          
                                      mis-scored.       

  F11               Lite mode         `APP_MODE=lite`   Must
                                      runs without      
                                      PyTorch or the    
                                      transformer       
                                      models, uses      
                                      under 1 GB RAM,   
                                      and delivers      
                                      every feature     
                                      above with        
                                      slightly lower    
                                      accuracy. A       
                                      banner shows the  
                                      active mode.      

  F12               Precomputed demo  The demo          Must
                    results           dataset's         
                                      analysis is saved 
                                      as a parquet file 
                                      in the repo and   
                                      loads instantly   
                                      in both modes,    
                                      keyed by file     
                                      hash.             

  F13               Input validation  Bad files produce Must
                                      a clear message,  
                                      never a crash     
                                      (see section 11). 

  F14               Compare versions  Aspect sentiment  Stretch
                    or periods        by version or     
                                      month, plus a     
                                      spike alert for   
                                      sudden jumps in   
                                      complaints.       

  F15               Local LLM polish  Optional toggle   Stretch
                                      to polish ticket  
                                      and reply wording 
                                      using a small     
                                      local model (full 
                                      mode, local       
                                      only).            
  -----------------------------------------------------------------------

## 4. How It Works (Pipeline)

1.  **Ingest and validate:** read file, map columns, drop empty rows,
    flag exact duplicates. If the file hash matches a precomputed
    result, load it and skip to step 10.
2.  **Normalize:** lowercase, collapse repeated letters ('bahuuut' to
    'bahut'), map Hinglish spelling variants, convert emojis to
    sentiment hints, redact phone numbers and emails.
3.  **Language tag:** English, Hindi (Devanagari), Hinglish (romanized)
    or mixed.
4.  **Clause split:** break on punctuation and contrast words (but,
    however, lekin, par, magar) so one review can hold several opinions.
5.  **Aspect tagging:** keyword seeds (English and Hinglish) first; in
    full mode, embedding similarity for clauses the seeds miss.
6.  **Clause sentiment:** full mode blends transformer score with the
    Hinglish-aware lexicon; lite mode uses the lexicon alone. Sarcasm
    and rating-conflict checks applied.
7.  **Safety scan:** pattern and context check on every clause
    (embedding check in full mode only).
8.  **Grouping:** full mode clusters sentence embeddings; lite mode
    clusters TF-IDF vectors. Each cluster is labeled.
9.  **Impact and ranking:** regression of star rating on aspect
    sentiment, then priority scores.
10. **Actions and UI:** dashboard, drill-down, simulator, tickets,
    replies, exports.

## 5. Data Contract

Agree on this in the first 30 minutes. Everyone codes against it.

**Entry point:** `analyze(df, mode) -> Results`, where `Results` holds
the four tables below plus a `meta` dict (mode, row counts, warnings,
timings).

  -----------------------------------------------------------------------
  Table                               Columns
  ----------------------------------- -----------------------------------
  `reviews`                           `review_id`, `text`, `rating`
                                      (nullable), `date` (nullable),
                                      `version` (nullable), `lang` (en,
                                      hi, hinglish, mixed),
                                      `is_duplicate`, `rating_source`
                                      (given or pseudo)

  `clauses`                           `review_id`, `clause_id`,
                                      `clause_text`, `aspect`, `score`
                                      (-1 to 1), `polarity` (neg, neu,
                                      pos), `confidence`,
                                      `sarcasm_suspect`,
                                      `rating_conflict`, `safety_tier`
                                      (none, medium, high, critical),
                                      `cluster_id`

  `clusters`                          `cluster_id`, `aspect`, `polarity`,
                                      `label`, `example_clause_id`,
                                      `size`

  `insights`                          `aspect`, `polarity`, `mentions`,
                                      `pct_negative`, `beta`, `ci_low`,
                                      `ci_high`, `damage`,
                                      `priority_rank`, `confidence_label`
                                      (ok or low), `single_cause_unhappy`
  -----------------------------------------------------------------------

**Thresholds** (in `config/settings.yaml`, tunable): polarity is
negative at score -0.25 or below and positive at 0.25 or above; minimum
mentions for a confident estimate is 30; cluster distance threshold 0.4;
upload cap 2,000 rows in lite mode and 1,000 for live full-mode runs.

## 6. Tech Stack and Why

### Model reality check

I checked the candidate models before choosing:

-   **XLM-T sentiment
    (`cardiffnlp/twitter-xlm-roberta-base-sentiment`):** three-class
    (negative, neutral, positive), pre-trained on about 198M tweets and
    fine-tuned for sentiment on 8 languages including Hindi and English.
    It handles informal, emoji-heavy text well. Its reported score on
    its own multilingual tweet benchmark is about 69% macro F1: decent,
    not perfect. Romanized Hinglish is not among its 8 training
    languages, so we never trust it alone.
-   **HingBERT family (`l3cube-pune/hing-*`):** trained on romanized
    Hindi-English code-mixed text, but these are base language models,
    not ready-made sentiment classifiers. They need fine-tuning on
    labeled Hinglish data we do not have. Skipped.
-   **Embeddings (`paraphrase-multilingual-MiniLM-L12-v2`):** 118M
    parameters, 384-dimensional vectors, 50+ languages including Hindi,
    Apache-2.0 license, built for clustering and similarity. Small and
    fast on CPU.

**Decision:** full mode blends three signals: transformer sentiment, a
Hinglish-extended lexicon (VADER with added Hinglish words), and the
star rating as a sanity check. Starting blend: 0.5 transformer, 0.35
lexicon, 0.15 rating prior. Lite mode uses lexicon plus rating prior
only (re-tuned weights). Final weights come from the benchmark.

### Two modes

  -------------------------------------------------------------------------
                          Full mode                 Lite mode
  ----------------------- ------------------------- -----------------------
  Where it runs           Your laptop (live demo,   Free hosted link
                          offline)                  

  Sentiment               XLM-T + Hinglish          Hinglish lexicon +
                          lexicon + rating prior    rating prior

  Aspect matching         Seeds + embeddings        Seeds only

  Grouping                MiniLM embeddings +       TF-IDF (word and
                          agglomerative clustering  character n-grams) +
                                                    agglomerative
                                                    clustering

  Safety                  Patterns + embedding      Patterns + context
                          prototypes                guard

  Memory                  about 3 to 4 GB           under 1 GB target

  Requirements file       `requirements-full.txt`   `requirements.txt`
  -------------------------------------------------------------------------

Character n-grams in lite mode help with Hinglish spelling variants
without any model. If the full models fail to load, the app falls back
to lite mode and says so.

### Application stack

  ---------------------------------------------------------------------------------------------------
  Layer             Choice                                    Purpose           Fallback / constraint
  ----------------- ----------------------------------------- ----------------- ---------------------
  UI design         Google Stitch                             Generate and      Stitch is not part of
                                                              refine screen     runtime; implement
                                                              concepts, layout  designs in React
                                                              and design system 

  Frontend          React + TypeScript + Vite                 Responsive        Keep screens reusable
                                                              dashboard,        and API-driven
                                                              uploads, filters, 
                                                              issue drill-down, 
                                                              charts, simulator 
                                                              and exports       

  Styling /         Tailwind CSS + reusable React components  Consistent visual Avoid overbuilding a
  components                                                  system and fast   component library
                                                              iteration         

  Charts            Recharts                                  Sentiment, issue  Use accessible labels
                                                              ranking, language and tables alongside
                                                              mix and simulator charts
                                                              before/after      
                                                              charts            

  Backend           Python 3.11 + FastAPI                     REST API, upload  Host on a
                                                              validation,       Python-capable
                                                              analysis          service; verify
                                                              orchestration,    free-tier limits
                                                              simulator,        
                                                              tickets and       
                                                              exports           

  Tabular           pandas + NumPy                            CSV/XLSX parsing  Cap rows and file
  processing                                                  and review-level  size on hosted Lite
                                                              tables            mode

  Database          Supabase PostgreSQL                       Persist analysis  No database is needed
                                                              metadata, status, for model execution
                                                              summary metrics   itself; if Supabase
                                                              and saved         is unavailable during
                                                              report/result     local development,
                                                              payloads          keep an optional
                                                                                in-memory development
                                                                                mode

  Sentiment (full)  XLM-T via Transformers + PyTorch CPU,     Multilingual      Lite mode
                    dynamic int8 quantization, batch 32, max  sentiment signal  
                    length 64                                                   

  Hinglish layer    VADER lexicon extended with \~300         Covers romanized  Expand lexicon from
                    Hinglish words, negations, intensifiers   Hindi-English     benchmark errors
                    and spelling normalization                code-mixing and   
                                                              powers Lite mode  

  Embeddings (full) sentence-transformers                     Semantic          TF-IDF in Lite mode
                    `paraphrase-multilingual-MiniLM-L12-v2`   similarity and    
                                                              clustering        

  Clustering        scikit-learn AgglomerativeClustering +    Group similar     Keep clustering
                    TF-IDF labels                             comments          bounded for hosted
                                                                                uploads

  Impact model      statsmodels OLS with robust standard      Estimate rating   Difference-in-means
                    errors                                    association and   fallback
                                                              confidence        
                                                              intervals         

  Safety            Regex/lexicon, context and negation       Auditable hazard  Safety terms and
                    guards; embeddings in Full mode           escalation        thresholds are
                                                                                configurable

  Actions           Jinja2 templates; optional local Ollama   Ticket/reply      Templates are the
                    polish in Full mode                       generation        reliable default
                                                              without paid APIs 

  Tests             pytest + frontend tests                   Protect API       Run smoke tests
                                                              contracts and     before deployment
                                                              core behaviors    

  Hosting           Vercel frontend + Python-capable FastAPI  Public demo with  Confirm current
                    host + Supabase                           persistent        free-tier limits and
                                                              metadata          cold-start behavior
                                                                                before relying on
                                                                                them
  ---------------------------------------------------------------------------------------------------

**Mode strategy:** `APP_MODE=lite` avoids PyTorch and transformer
downloads and is the hosted default. `APP_MODE=full` is for local
demonstration after models have been downloaded. Keep the same API
response schema in both modes.

**Privacy defaults:** uploaded review contents are processed for
analysis and are not saved as raw files by default. Persist only what is
needed to reopen a result (analysis metadata and result summaries); if
complete review-level drill-down must persist, store the minimum
necessary fields and disclose that behavior in the UI. Redact email
addresses and phone numbers from visible exports.

**Deliberately excluded:** paid LLM APIs, model fine-tuning, vector
databases, large local LLM inference on every review, and mandatory paid
hosting. Supabase is the relational store; a separate vector database is
unnecessary for this prototype.

## 7. Key Algorithms

### 7.1 Fix-This-First Simulator (headline feature)

-   Build a review-by-aspect table: for each review with a rating,
    whether each aspect is mentioned negatively or positively.
-   Fit
    `rating = a + sum(b_aspect x negative_aspect) + sum(g_aspect x positive_aspect)`
    using OLS with robust errors. Each `b` is the rating points lost
    when that aspect is complained about. Aspects that move together too
    closely (very high correlation) are merged or one is dropped, so
    estimates stay stable.
-   **Rating damage** for an aspect = number of negative reviews x
    \|b\|. **Priority score** = rating damage, shown alongside the plain
    complaint count.
-   **Projected rating** when a fraction `f` of an aspect's complaints
    are fixed = current average + f x (negative reviews / total reviews)
    x \|b\|, capped at 5.0. Effects of several slider moves are added;
    interactions are ignored and the UI says so.
-   **Unhappy reviews that would disappear:** count of reviews rated 2
    stars or less whose only negative aspect is the one being fixed.
-   **Honesty guardrails:** aspects with fewer than 30 mentions are
    marked 'low confidence'; a tooltip says the estimate is an
    association, not a guaranteed cause. If no rating column exists, a
    pseudo-rating from sentiment is used, and the simulator shows a
    visible label 'sentiment-based estimate' because that estimate is
    circular.
-   **Slider UI:** one slider per top-5 issue, plus a 'fix top 3'
    preset; the projected average rating and a before/after bar update
    instantly.

### 7.2 Safety flag

-   Three tiers. **Critical:** fire, explosion, sparks, shock, burn,
    smoke. **High:** allergic reaction, rash, choking, bleeding, injury,
    fainting. **Medium:** overheating, swelling battery, leakage.
-   Terms in English, Hinglish ('aag lag gayi', 'garam ho gaya',
    'current laga') and Devanagari.
-   **Negation guard** looks a few words around the hit ('no
    overheating', 'koi problem nahi').
-   **Context guard:** the hazard word must co-occur with a product,
    body or event context (and, in full mode, may also match a hazard
    prototype by embedding similarity), so 'burn rate' does not trigger.
-   Output: red banner pinned above all rankings, each case with
    severity, review text and suggested action ('escalate to QA and
    Compliance within 24 hours'). Safety cases are never ranked by
    frequency.

### 7.3 Hinglish support

-   **Normalizer:** repeated letters, spelling variants (kharab,
    kharaab, kharb), short forms (nhi, nahi, ni), emoji mapping.
-   **Lexicon:** about 300 entries of positive, negative, negation and
    intensifier words ('bakwaas', 'ghatiya', 'bekaar', 'zabardast',
    'paisa vasool', 'bahut', 'bilkul'), loaded into the VADER lexicon at
    startup.
-   **Aspect seeds** in Hinglish ('awaaz' for sound, 'dabba' for
    packaging, 'kharab hona' for failure).
-   **Language badge** on every review so the panel can see the system
    recognised the language.
-   Lexicon lives in a plain text file so teammates can extend it fast.

### 7.3a Indian Languages (Kannada, Tamil, Telugu, Hindi)

**Added after v3.** The original v3 covered only English, Hindi and
Hinglish. This section extends it to Kannada, Tamil and Telugu written
the way people actually type them: **local-language words in English
letters** ('romanized' text).

**The problem:** most reviews look like these. The English nouns are
easy; the opinion is in the local language. Examples below are to be
verified by a native speaker on the team.

  -----------------------------------------------------------------------
  Style                   Example                 Meaning
  ----------------------- ----------------------- -----------------------
  Hinglish                'battery bakwaas hai'   the battery is rubbish

  Kanglish (Kannada)      'camera tumba           the camera is very good
                          chennagide'             

  Tanglish (Tamil)        'phone romba waste'     the phone is a big
                                                  waste

  Tenglish (Telugu)       'camera chala bagundi'  the camera is very good
  -----------------------------------------------------------------------

**Honest model check:** XLM-T's sentiment fine-tuning covers 8 languages
(Arabic, English, French, German, Hindi, Italian, Spanish, Portuguese).
The MiniLM embedding model's published language list includes Hindi,
Gujarati, Marathi and Urdu but not Kannada, Tamil or Telugu. Neither is
validated for romanized Indian languages, so we do **not** claim
model-based understanding for them.

**Approach: language packs.** The same code runs for every language;
each language is a folder:

-   sentiment lexicon (positive, negative, negation, intensifiers) and a
    spelling-variant map
-   aspect seeds and hazard terms in that language
-   function words used for language detection
-   30 hand-labeled test clauses

Adding a language means adding a folder, not new code. Packs: Hinglish
(hours 1 to 3 as planned), then Kanglish, Tanglish and Tenglish written
in parallel by teammates who speak them.

**Why this works without a model:**

-   Code-mixed reviews are mostly **English nouns** (battery, delivery,
    camera, price) plus local sentiment words. Aspect detection already
    works on the English nouns; only the sentiment words need the local
    lexicon, which is far smaller than a full vocabulary.
-   Character n-gram TF-IDF clustering groups spelling variants
    ('bagundi', 'bagundhi') without any language model.
-   **Negation differs by language.** In Kannada, Tamil and Telugu the
    negation word usually comes after the opinion word ('illa', 'ledu',
    'alla'), so the negation window looks in both directions.

**Native script:** Hindi in Devanagari goes through XLM-T in full mode.
Kannada, Tamil and Telugu script is best effort: keep the top 50
sentiment words in both scripts, and optionally transliterate to Roman
letters with an open-source library if it installs cleanly in Hour 0.
Not promised.

**Language detection:** count matches against each pack's function words
and lexicon; the best ratio above a threshold wins, otherwise English.
The review badge shows it (for example 'Tanglish').

**What we claim in the demo (never 'all Indian languages'):**

  -----------------------------------------------------------------------
  Tier                    Languages               Status
  ----------------------- ----------------------- -----------------------
  1                       English, Hinglish       Fully tested, accuracy
                                                  measured

  2                       Kanglish, Tanglish,     Working through
                          Tenglish (only those    language packs,
                          with a native speaker   accuracy measured on 30
                          on the team)            clauses each and
                                                  reported as is

  3                       Any other language      Architecture-ready: add
                                                  a pack
  -----------------------------------------------------------------------

**Effort and schedule:** each extra pack takes about 60 to 90 minutes
with a native speaker: 150 to 200 lexicon entries, spelling variants, 30
test clauses. Write them in hours 3 to 6, in parallel with the
simulator, by a different person. If nobody on the team speaks a
language, drop that language instead of guessing.

**UI additions:** language badge on every review, a language mix chart
on the Overview, a language filter in the drill-down, and a 'supported
languages' note in the sidebar.

**Demo data:** add 15 reviews per extra language to `demo_hinglish.csv`,
covering the same planted aspects (battery, delivery) so clusters and
the simulator include them.

**Metrics:** per-language accuracy table on the 30-clause sets (expect
lower than English; report honestly), and language detection correct on
at least 80% of 20 test reviews per language.

**Optional stretch (local only):** a small multilingual local model
judges only clauses the lexicon is unsure about, capped at about 200
clauses because it is slow on CPU, and never on the hosted app. Test in
Hour 0 whether a candidate model really handles romanized Kannada, Tamil
and Telugu before relying on it.

**Panel answer:** 'Do you support all Indian languages?' We support
English, Hinglish and every language pack we have built, each measured
separately. Adding another language is a folder of word lists, not new
code.

**Risk:** a weak lexicon gives wrong sentiment in that language.
Mitigation: native-speaker review of every pack, per-language accuracy
shown openly, and the rating prior still anchors sentiment when the text
is unclear.

### 7.4 Sarcasm and ambiguity

-   **Rule 1:** a positive phrase ('great', 'wow', 'thanks', 'waah')
    followed by a failure event ('broke', 'stopped working', 'late',
    'toot gaya') in the same or next clause is flagged as suspected
    sarcasm and scored negative.
-   **Rule 2:** text sentiment strongly positive but rating 1 to 2 stars
    (or the reverse) is flagged as a conflict and shown with a warning
    badge.
-   Flagged items are counted using the rating-aligned polarity and
    listed in a 'needs review' view. Optional: ask the local LLM to
    judge flagged items only.

### 7.5 Grouping and labels

-   Cluster separately for each aspect and polarity using agglomerative
    clustering (cosine distance, threshold from settings, tuned around
    0.35 to 0.45). Vectors are MiniLM embeddings in full mode and TF-IDF
    in lite mode.
-   Merge clusters smaller than 3 into 'other'.
-   Label each cluster with its top TF-IDF terms and show the clause
    nearest the center as the example.
-   Clauses that match no aspect seed are clustered together and shown
    as 'Emerging topics', which also proves the system is not limited to
    a fixed list.
-   Aspect seeds live in a YAML file, so the same system works for other
    product types.

### 7.6 Auto-action output

-   **Routing map:** Battery, Performance, Quality, Design go to
    Product/Engineering. Delivery and Packaging go to Logistics. Support
    goes to Customer Support. Price goes to Pricing/Marketing. Safety
    goes to QA/Compliance with priority P0.
-   **Ticket fields:** title, team, priority, one-paragraph summary with
    stats, estimated rating uplift, 3 to 5 evidence quotes with review
    IDs, suggested next steps, success metric ('negative mentions down
    50% in 30 days').
-   **Reply drafts:** empathetic templates per aspect in English and
    Hinglish, no false promises, copy button.
-   **Export:** one ticket as Markdown, all tickets as CSV.
-   **Optional:** local LLM polish toggle, off by default and local
    only.

## 8. Deployment, Database and Git

### Deployment architecture

1.  **Design:** use Google Stitch to draft the dashboard, upload flow,
    issue drill-down, simulator, and actions screens. Treat generated
    screens as design references; rebuild them as responsive React
    components.
2.  **Frontend:** deploy the Vite/React app to Vercel.
3.  **Backend:** deploy FastAPI to a Python-capable host that supports
    the required dependencies and request duration. Hosted mode defaults
    to Lite NLP. Confirm free-tier RAM, CPU, request timeout, upload
    limit, cold starts and sleeping behavior before selecting the
    provider.
4.  **Database:** use Supabase PostgreSQL for persistent analysis
    records and saved result metadata.
5.  **Local demo:** run Full mode locally after downloading model files.
    Keep a precomputed demo analysis so the demo remains reliable if a
    model download or hosting service fails.
6.  **Backup:** record the demo and keep a known-good demo CSV.

### Initial database schema

Keep the schema small and avoid storing raw uploads permanently by
default.

-   `analyses`: `id` (UUID primary key), `created_at`, `source_name`
    (sanitized filename), `mode` (`lite`/`full`), `status`,
    `review_count`, `average_rating`, `negative_rate`,
    `language_summary` (JSONB), `top_issues` (JSONB), `warnings`
    (JSONB), `processing_ms`, `result_version`.
-   `analysis_reviews` (optional, only if persistent drill-down is
    required): `id`, `analysis_id` (foreign key), `review_key`,
    `redacted_text`, `rating` (nullable), `language`, `is_duplicate`. Do
    not store original email addresses, phone numbers or uploaded file
    bytes.
-   `saved_actions` (optional): `id`, `analysis_id`, `issue_key`,
    `ticket_json` (JSONB), `reply_draft`, `created_at`.

For the first working version, `analyses` can store summary results and
a compact JSONB payload. Add `analysis_reviews` only if users must open
evidence after a new session. Apply Supabase Row Level Security (RLS)
before adding user accounts or exposing user-specific records. If there
is no authentication, do not pretend records are private per user; use
demo-safe data and restrict write access through the backend service
role, never the browser.

### API contract (initial)

-   `GET /health` --- backend health and active mode.
-   `POST /api/analyze` --- multipart CSV/XLSX upload; returns
    `analysis_id`, summary, top issues, safety alerts, and warnings.
-   `GET /api/analyses` --- list recent saved analyses.
-   `GET /api/analyses/{analysis_id}` --- retrieve saved
    summary/results.
-   `GET /api/analyses/{analysis_id}/reviews` --- evidence drill-down,
    only if review evidence is persisted for that analysis.
-   `POST /api/analyses/{analysis_id}/simulate` --- calculate projected
    rating from selected issue fix percentages.
-   `POST /api/analyses/{analysis_id}/actions` --- generate a ticket and
    reply draft.
-   `GET /api/analyses/{analysis_id}/export?format=csv|md` --- download
    supported exports.
-   `GET /health` must not reveal secrets or connection strings.

Use Pydantic request/response schemas, file-size and row-count limits,
CORS restricted to the deployed frontend origin, and clear error
responses. Keep database credentials server-side in environment
variables. Never put the Supabase service-role key in the Vite frontend.

### Git repo checklist

-   Clear README with features, screenshots, setup, environment
    variables, API instructions, both modes, hosted link and
    limitations.
-   `.gitignore` for `.env`, model weights, caches, virtual
    environments, uploads, build artifacts and local database files.
-   `frontend/` for React app; `backend/` for FastAPI and NLP pipeline;
    `tests/` for backend and frontend tests.
-   `requirements.txt` for Lite backend; `requirements-full.txt` for
    local Full mode.
-   `data/demo_hinglish.csv` and a compact precomputed demo result
    fixture.
-   `.env.example` with placeholder names only, never secrets.
-   Pin dependencies that pass the clean install test.

### Deployment verification

1.  Run the Lite backend locally with only Lite dependencies.
2.  Run the frontend against the local FastAPI URL.
3.  Test upload, analysis, simulator, drill-down, export and database
    persistence.
4.  Deploy the backend and set CORS to the exact Vercel frontend origin.
5.  Add backend URL and public Supabase URL/key (only if using a
    client-side Supabase feature) to Vercel environment settings. The
    Supabase service-role key belongs only on the backend.
6.  Deploy the frontend to Vercel; verify fresh upload and reload of
    saved analysis.
7.  Verify database table policies, timeouts, row limits, and that
    secrets do not appear in browser bundles or logs.

## 9. Demo Data Recipe

**Input schema:** `review_text` (required); `rating`, `date`, `version`,
`review_id` (optional).

**File 1: real data.** A public review set of 5,000 to 10,000 rows from
one product category (candidates: an Indian e-commerce review dataset
from Kaggle or an Amazon reviews category; confirm license and
availability in Hour 0). Precompute results once and save them. Real
data shows the system is not rigged.

**File 2: `demo_hinglish.csv`, 300 reviews built by hand and template.**
Needed because real datasets rarely contain Hinglish, safety cases or
clean sarcasm. Recipe:

1.  Pick one product (for example wireless earbuds) and 8 aspects.
2.  Write 5 to 8 phrasings per aspect and polarity, including 40
    Hinglish ones, then combine them into reviews of 1 to 3 clauses.
3.  **Generate ratings from a known rule so the simulator has something
    real to find:** base rating 4.4, minus 1.2 if battery is negative,
    minus 0.8 if delivery is negative, minus 0.5 if packaging is
    negative, plus noise, rounded and clipped to 1 to 5.
4.  Plant: a clear battery cluster, a delivery cluster, 5 sarcasm cases,
    10 mixed-opinion reviews, 3 safety cases, and two versions (v1, v2)
    where v2 fixes delivery.
5.  **Validation test:** the regression must recover the planted effects
    within about 0.3 rating points. If it does not, the simulator math
    or the aspect tagging has a bug.

Be upfront in the demo that file 2 is partly synthetic, built to show
edge cases.

## 10. Screens

1.  **Upload:** drop file or pick a bundled demo, column-mapping
    preview, Analyze button with progress, mode banner (Full or Lite),
    upload limits and privacy note.
2.  **Overview:** safety banner (if any), metric tiles (reviews, average
    rating, % negative), top issues ranked with rating damage, mentions
    and % negative, and a top-positives strip.
3.  **Issue drill-down:** clusters under the issue, example sentences,
    the actual reviews with highlighted clauses, language and sarcasm
    badges.
4.  **Simulator:** sliders, projected rating, before/after chart,
    'single-cause unhappy reviews' count.
5.  **Actions:** ticket preview, reply drafts, export buttons.
6.  **History:** list prior saved analyses with date, filename label,
    review count, mode and status; open or delete saved metadata.
7.  **Compare (stretch):** version or month selector, per-aspect
    sentiment deltas and spike alerts.
8.  **Database / service states:** loading, empty, offline, permission
    and persistence errors must be visible and recoverable.

## 11. Edge Cases and Error Handling

  -----------------------------------------------------------------------
  Case                                Handling
  ----------------------------------- -----------------------------------
  Mixed opinions in one review        Clause split; each aspect scored
                                      separately

  One review, several aspects         Counted once per aspect

  Sarcasm                             Positive phrase plus failure event
                                      rule; rating conflict check;
                                      warning badge

  Negation ('not bad', 'nahi kharab') Lexicon negation handling plus
                                      transformer score in full mode

  Hinglish and Devanagari             Normalizer, Hinglish lexicon,
                                      language badge

  Very short reviews ('ok', 'nice')   Low confidence; excluded from
                                      clustering

  Duplicates and spam                 Exact duplicates collapsed;
                                      repeated near-identical text
                                      down-weighted

  Missing rating or date              Pseudo-rating from sentiment
                                      (labeled); time views hidden

  Mentions a hazard but negated       Negation guard stops false alarms

  Very long reviews                   Truncate each review to 1,000
                                      characters before analysis

  File encoding problems              Try UTF-8, then UTF-8 with BOM,
                                      then Latin-1; show a message if all
                                      fail

  No text column found                Ask the user to pick one from a
                                      dropdown

  Empty file, one-row file, all-blank Clear message, no crash
  text                                

  Ratings out of range or non-numeric Coerce to 1 to 5, drop invalid,
                                      warn with the count

  Hosted app gets a very large upload Lite mode caps uploads (2,000 rows)
                                      with a clear message

  Personal data in reviews            Phone numbers and emails redacted
                                      in the UI, tickets and exports; raw
                                      files are not stored permanently by
                                      default. Analysis metadata/results
                                      are saved to Supabase when
                                      connected

  Full models fail to load            Fall back to lite mode with a
                                      banner
  -----------------------------------------------------------------------

## 12. Test Plan, Metrics and Acceptance Checklist

**Hour 0 benchmark (the gate):** hand-label 100 clauses (60 English, 40
Hinglish) as positive, neutral or negative. Compare XLM-T alone, lexicon
alone, and the blend. Keep the blend if it reaches at least 75% accuracy
overall; if Hinglish is weaker, raise the lexicon weight. Run the same
set in lite mode to know how much accuracy lite gives up. Re-run after
the Hinglish polish.

**Five automated tests (pytest):** (1) clause splitter on 'Great camera,
terrible battery'; (2) safety negation ('no overheating' must not
trigger, 'phone caught fire' must); (3) sentiment sign on 10 known
sentences; (4) simulator math on a tiny hand-computed example; (5)
end-to-end smoke test on the demo file.

  -----------------------------------------------------------------------
  Metric                              Target
  ----------------------------------- -----------------------------------
  Clause sentiment accuracy on        at least 75% (full), at least 70%
  labeled set                         (lite)

  Safety recall on 40 test sentences  at least 90% (precision at least
                                      80%)

  Aspect split on 'Great camera,      correct on at least 8 of 10
  terrible battery'-type reviews      

  Simulator recovers planted effects  within about 0.3 rating points

  Processing speed on CPU (full mode) 500 reviews in under 90 seconds;
                                      larger demo set precomputed and
                                      cached

  Lite mode memory                    under 1 GB while analysing 500
                                      reviews

  Hosted app first load               precomputed demo visible in under
                                      10 seconds

  Simulator                           slider updates in under 1 second
  -----------------------------------------------------------------------

**Acceptance checklist: 'fully working' means all of these are true.**

1.  Upload a CSV and get ranked issues without errors, in both modes.
2.  Every issue opens its supporting reviews with the relevant clause
    highlighted.
3.  A review with mixed opinions is split and scored per aspect.
4.  A Hinglish review is recognised, badged and scored sensibly.
5.  A sarcastic review is flagged with a warning badge.
6.  Safety cases appear in a red banner above all rankings.
7.  Moving a slider changes the projected rating live.
8.  A ticket and a reply draft can be created and exported.
9.  A bad file (wrong encoding, empty, no text column) shows a message
    instead of crashing.
10. The hosted link loads the precomputed demo, handles a fresh upload,
    and can reload saved analysis metadata from Supabase.
11. The five pytest tests pass.
12. The README lets a stranger run it in under 10 minutes.
13. The demo script runs twice in a row, offline, without errors.

## 13. 10-Hour Build Plan

Build lite first: a complete product by hour 3, then upgrade it.
Whatever happens later, there is always something to demo.

  -----------------------------------------------------------------------
  Hours                   Work                    Milestone / go-no-go
  ----------------------- ----------------------- -----------------------
  0 to 1                  Set up environment and  Contract agreed;
                          repo, agree the data    install works on every
                          contract, choose and    laptop
                          download datasets,      
                          hand-label the          
                          100-clause benchmark,   
                          start the demo data     
                          recipe                  

  1 to 3                  **Lite vertical slice   **M1 (hour 3): upload a
                          end to end:** ingest,   CSV, see ranked issues,
                          normalize, language     click to see reviews.**
                          tag, clause split,      If not met, cut stretch
                          aspect seeds, Hinglish  items now
                          lexicon sentiment,      
                          TF-IDF grouping,        
                          ranking, React          
                          dashboard with          
                          drill-down              

  3 to 4.5                Add full mode: XLM-T    **M2:** full mode beats
                          sentiment blend, MiniLM lite by a clear margin
                          embeddings, `APP_MODE`  or the default stays
                          switch, benchmark       lite for the live demo
                          numbers, pick final     
                          weights                 

  4.5 to 6                Regression, rating      **M3:** simulator
                          damage, projected       recovers planted
                          rating, sliders,        effects
                          validation test on      
                          synthetic data          

  6 to 7                  Safety lexicon and      **M4:** safety recall
                          guards, sarcasm rules,  target met
                          Hinglish lexicon        
                          expansion, re-run       
                          benchmark               

  7 to 8                  Ticket and reply        **M5 and feature freeze
                          templates, routing map, at hour 8.** Only fixes
                          exports, UI polish,     after this
                          input validation        

  8 to 9                  Precompute demo         **M6:** public link
                          results, lite-mode      live
                          memory check,           
                          Dockerfile, README,     
                          push to GitHub, deploy  

  9 to 10                 Run the acceptance      All 13 checklist items
                          checklist, fix bugs,    true
                          rehearse twice, record  
                          backup video            
  -----------------------------------------------------------------------

**If behind, cut in this order:** local LLM polish, version and time
comparison, Hinglish reply templates (keep English), sarcasm Rule 1
(keep the rating-conflict flag), Docker (keep the selected free Python
backend host).

**Never cut:** sentiment, topics, grouping, ranking, traceability,
simulator, safety flag, precomputed demo results, input validation.

## 14. Risks and Mitigations

  -----------------------------------------------------------------------
  Risk                                Mitigation
  ----------------------------------- -----------------------------------
  Transformer weak on romanized       Blend with Hinglish lexicon and
  Hinglish                            rating prior; benchmark early

  Model download or loading problems  Lite-first build means a working
                                      product already exists; fallback to
                                      lite with a banner

  Slow CPU inference                  int8 quantization, short max
                                      length, batching, cached demo
                                      results, Colab for precompute

  Hosted app runs out of memory       Lite mode by default on the host,
                                      upload cap, precomputed demo, no
                                      model files in the repo

  Free hosting limits or policy       Confirm limits in Hour 0; keep
  change                              Docker and video backup; live demo
                                      runs locally

  Lite mode accuracy noticeably lower Measure the gap; say so openly;
                                      full mode is for the live demo

  Simulator shows nonsense            Validation test on synthetic data;
                                      minimum 30 mentions; confidence
                                      labels

  Clusters look messy                 Tune threshold on the demo set,
                                      merge tiny clusters, label from
                                      TF-IDF and example clause

  Safety false alarms                 Negation and context guards, test
                                      set of 40 sentences

  Scope too big for the team          Milestones with go/no-go checks,
                                      cut list, feature freeze at hour 8

  Teammates blocked on each other     Data contract agreed in Hour 0
  -----------------------------------------------------------------------

## 15. Demo Script and Panel Q&A

**Script (5 minutes)**

1.  **Problem (30 s):** thousands of reviews, nobody reads them all.
2.  **Upload (30 s):** drop the dataset; show the progress bar and the
    mode banner.
3.  **Overview (60 s):** point at the red safety banner first, then the
    ranked issues with rating damage.
4.  **Drill-down (60 s):** open the battery issue; show clusters
    including a Hinglish review with its language badge; show a flagged
    sarcastic review.
5.  **Simulator (60 s):** drag the battery slider and watch the
    projected rating climb; mention single-cause unhappy reviews.
6.  **Action (30 s):** click 'Create ticket' and show the reply draft.
7.  **Close (30 s):** repeat the pitch line and share the public link
    and GitHub repo.

**Likely panel questions**

  -----------------------------------------------------------------------
  Question                            Answer
  ----------------------------------- -----------------------------------
  Why not just ask a big LLM?         It costs money per review, sends
                                      customer data to a third party, and
                                      is hard to explain. Our prototype
                                      avoids paid APIs, supports local
                                      Full mode, and links insights to
                                      supporting reviews; hosted requests
                                      use the deployed backend and
                                      configured database.

  Is the rating impact causal?        No. It is an association estimate
                                      with confidence labels, built to
                                      prioritise, and the UI says so.

  How accurate is it?                 We measured it on a hand-labeled
                                      set: show the numbers from the
                                      benchmark, for both modes.

  Does it handle other languages?     Embeddings cover 50+ languages and
                                      the lexicon is a plain file anyone
                                      can extend; Hinglish is our proof
                                      of that.

  Will it scale to 100,000 reviews?   Run it as a batch job with cached
                                      embeddings (GPU optional); the app
                                      shows precomputed results. We
                                      tested up to the demo size.

  What about fake reviews?            Exact duplicates are collapsed and
                                      near-identical spam is
                                      down-weighted; a fuller fake-review
                                      filter is a next step.

  Why two modes?                      Full mode for accuracy on a laptop;
                                      lite mode so anyone can open a free
                                      public link.
  -----------------------------------------------------------------------

## 16. Repo Layout

    frontend/               # React + TypeScript + Vite + Tailwind
      src/
        components/         # reusable UI components
        pages/              # Upload, Overview, Drill-down, Simulator, Actions, History
        lib/api.ts           # typed FastAPI client
        types/               # API response types
    backend/
      main.py                # FastAPI app and routes
      pipeline/
      ingest.py normalize.py clauses.py aspects.py
      sentiment.py safety.py cluster.py impact.py actions.py
    backend/config/
      settings.yaml         # thresholds, weights, caps
      aspects.yaml          # aspect seeds (English + Hinglish)
      hinglish_lexicon.txt  # words, negations, intensifiers
      safety_terms.yaml     # tiered hazard terms
      routing.yaml          # aspect to team map
    backend/templates/      # ticket and reply templates
    data/                   # demo_hinglish.csv and compact demo results
    scripts/download_models.py
    tests/                  # pytest API, pipeline, safety and simulator tests
    .env.example
    backend/Dockerfile
    backend/requirements.txt        # lite
    backend/requirements-full.txt   # full/local
    frontend/package.json
    frontend/vite.config.ts
    frontend/tailwind.config.ts
    .gitignore
    README.md

## 17. v4 Change Log and Open Checks

### Confirmed decisions

-   Product name: **Revora AI** (working name; trademark/domain
    availability not verified).
-   Design tool: Google Stitch.
-   Frontend: React + TypeScript + Vite + Tailwind CSS; charts with
    Recharts.
-   Backend: Python + FastAPI.
-   Database: Supabase PostgreSQL for saved analysis metadata and result
    history.
-   Hosting target: Vercel for frontend; a compatible Python host for
    FastAPI; Supabase for PostgreSQL.
-   No paid APIs. Lite mode is the hosted default; Full mode is the
    local demo mode.
-   Preserve the core features: aspect sentiment, semantic grouping,
    issue ranking, evidence traceability, Fix-This-First simulator,
    safety alerts, language packs, ticket/reply generation, exports,
    ambiguity flags, input validation, and precomputed demo.

### Verify before implementation

-   Check current free-tier limits for the chosen FastAPI host and
    Supabase plan.
-   Decide whether to persist redacted review-level evidence or only
    summaries. Persistent drill-down after reload requires storing
    review evidence.
-   Confirm whether authentication is needed. It is not included in the
    initial scope.
-   Verify the exact dataset license and model package licenses before
    bundling.
-   Confirm supported file types and maximum upload size for the
    selected backend host.
-   Measure Lite/Full accuracy on the hand-labeled benchmark; never
    claim unmeasured accuracy.
-   Keep the 10-hour plan as a target, not a guarantee. If behind,
    preserve core analysis and the simulator first; history polish and
    optional comparison/LLM polish can be deferred.
