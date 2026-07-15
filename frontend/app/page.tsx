"use client";

import { useState } from "react";
import { predictRent, type PropertyFeatures, type PredictionResponse } from "@/lib/api";
import { config } from "@/lib/config";

const PROPERTY_TYPES = ["Apartment", "Villa", "Townhouse", "Penthouse"];
const CITIES = ["Dubai", "Abu Dhabi", "Sharjah", "Ajman", "Ras Al Khaimah", "Fujairah", "Umm Al Quwain", "Al Ain"];
const FURNISHING_OPTIONS = ["Unfurnished", "Furnished"];

const DEFAULTS: PropertyFeatures = {
  beds: 2,
  baths: 2,
  type: "Apartment",
  area_in_sqft: 1200,
  city: "Dubai",
  furnishing: "Unfurnished",
  location: "Dubai Marina",
};

export default function Home() {
  const [form, setForm] = useState<PropertyFeatures>(DEFAULTS);
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    setResult(null);
    setLoading(true);
    
    try {
      const prediction = await predictRent(form);
      setResult(prediction);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong.");
    } finally {
      setLoading(false);
    }
  }

  function update<K extends keyof PropertyFeatures>(key: K, value: PropertyFeatures[K]) {
    setForm((prev) => ({ ...prev, [key]: value }));
  }

  return (
    <main className="min-h-screen blueprint-grid">
      <div className="mx-auto max-w-6xl px-4 sm:px-6 py-12 sm:py-16 lg:py-24">
        <div className="mb-12 sm:mb-16">
          <p className="font-data text-[10px] sm:text-xs tracking-[0.2em] text-blueprint-line uppercase mb-4">
            UAE Property Valuation &middot; Instant Estimate
          </p>
          <h1 className="font-display text-3xl sm:text-4xl lg:text-5xl xl:text-6xl font-semibold text-paper leading-[1.05] max-w-2xl">
            Estimate annual rent to the dirham.
          </h1>
          <p className="mt-4 sm:mt-5 text-blueprint-line/80 max-w-lg text-sm sm:text-base leading-relaxed">
            A model trained on 70,000+ real UAE listings estimates
            your property&apos;s market rent in seconds.
          </p>
        </div>

        <div className="grid lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)] gap-8 lg:gap-12 xl:gap-16 items-start">
          <div className="hidden lg:block sticky top-16">
            <BlueprintDiagram form={form} />
          </div>

          <div>
            <form onSubmit={handleSubmit} className="space-y-5 sm:space-y-6">
              <fieldset className="grid grid-cols-2 gap-3 sm:gap-4">
                <Field label="Bedrooms">
                  <input
                    type="number"
                    min={0}
                    max={15}
                    value={form.beds}
                    onChange={(e) => update("beds", Number(e.target.value))}
                    className="input"
                  />
                </Field>
                <Field label="Bathrooms">
                  <input
                    type="number"
                    min={0}
                    max={15}
                    value={form.baths}
                    onChange={(e) => update("baths", Number(e.target.value))}
                    className="input"
                  />
                </Field>
              </fieldset>

              <Field label="Property type">
                <select
                  value={form.type}
                  onChange={(e) => update("type", e.target.value)}
                  className="input"
                >
                  {PROPERTY_TYPES.map((t) => (
                    <option key={t} value={t}>{t}</option>
                  ))}
                </select>
              </Field>

              <Field label="Area (sqft)">
                <input
                  type="number"
                  min={200}
                  max={20000}
                  value={form.area_in_sqft}
                  onChange={(e) => update("area_in_sqft", Number(e.target.value))}
                  className="input"
                />
              </Field>

              <fieldset className="grid grid-cols-2 gap-3 sm:gap-4">
                <Field label="City">
                  <select
                    value={form.city}
                    onChange={(e) => update("city", e.target.value)}
                    className="input"
                  >
                    {CITIES.map((c) => (
                      <option key={c} value={c}>{c}</option>
                    ))}
                  </select>
                </Field>
                <Field label="Furnishing">
                  <select
                    value={form.furnishing}
                    onChange={(e) => update("furnishing", e.target.value)}
                    className="input"
                  >
                    {FURNISHING_OPTIONS.map((f) => (
                      <option key={f} value={f}>{f}</option>
                    ))}
                  </select>
                </Field>
              </fieldset>

              <Field label="Location / community">
                <input
                  type="text"
                  value={form.location}
                  onChange={(e) => update("location", e.target.value)}
                  placeholder="e.g. Dubai Marina"
                  className="input"
                />
              </Field>

              <button
                type="submit"
                disabled={loading}
                className="w-full font-display font-semibold text-blueprint-950 bg-gold hover:bg-gold/90 disabled:opacity-50 disabled:cursor-not-allowed transition-colors rounded-sm py-3.5 sm:py-4 mt-2 sm:mt-4 relative"
              >
                {loading ? (
                  <span className="flex items-center justify-center gap-3">
                    <span className="inline-block h-5 w-5 animate-spin rounded-full border-2 border-blueprint-950 border-t-transparent" />
                    Calculating…
                  </span>
                ) : (
                  "Generate valuation"
                )}
              </button>
            </form>

            {error && (
              <p className="mt-6 text-sm text-red-300 font-data">
                Couldn&apos;t generate a valuation: {error}
              </p>
            )}

            {result && <ValuationCertificate result={result} form={form} />}
            
            <div className="mt-12 pt-6 border-t border-blueprint-line/20">
              <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3">
                <div>
                  <p className="font-data text-xs text-blueprint-line/50 uppercase tracking-wider">
                    ML Engineer
                  </p>
                  <p className="font-display text-lg text-paper font-semibold">
                    Obed Yameogo
                  </p>
                  <p className="font-data text-sm text-blueprint-line/70">
                    TheHatBuddy
                  </p>
                </div>
                <a
                  href={config.websiteUrl}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="font-data text-sm text-gold hover:text-gold/80 transition-colors border border-gold/30 hover:border-gold/60 rounded-sm px-4 py-2 inline-flex items-center gap-2"
                >
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
                  </svg>
                  Visit my website
                </a>
              </div>
            </div>
          </div>
        </div>
      </div>
    </main>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <label className="block">
      <span className="font-data text-[10px] sm:text-xs tracking-wide text-blueprint-line/70 uppercase mb-1.5 block">
        {label}
      </span>
      {children}
    </label>
  );
}

function ValuationCertificate({
  result,
  form,
}: {
  result: PredictionResponse;
  form: PropertyFeatures;
}) {
  return (
    <div className="relative mt-8 bg-paper text-ink p-6 sm:p-8 rounded-sm">
      <CornerBrackets />
      <p className="font-data text-[10px] sm:text-xs tracking-[0.2em] uppercase text-ink/60 mb-3">
        Estimated Valuation
      </p>
      <p className="font-data text-3xl sm:text-4xl lg:text-5xl font-semibold tabular-nums break-words">
        {result.predicted_rent_aed.toLocaleString(undefined, { maximumFractionDigits: 0 })}
        <span className="text-base sm:text-lg ml-2 text-ink/60">{result.currency} / {result.frequency.toLowerCase()}</span>
      </p>
      <div className="mt-6 pt-6 border-t border-ink/15 grid grid-cols-2 gap-x-4 sm:gap-x-6 gap-y-2 font-data text-xs sm:text-sm text-ink/70">
        <span>{form.beds} bed &middot; {form.baths} bath</span>
        <span>{form.area_in_sqft.toLocaleString()} sqft</span>
        <span className="break-words">{form.type}, {form.furnishing.toLowerCase()}</span>
        <span className="break-words">{form.location}, {form.city}</span>
      </div>
    </div>
  );
}

function CornerBrackets() {
  const cls = "absolute w-3 h-3 sm:w-4 sm:h-4 border-blueprint-950/30";
  return (
    <>
      <span className={`${cls} top-2 left-2 border-t-2 border-l-2`} />
      <span className={`${cls} top-2 right-2 border-t-2 border-r-2`} />
      <span className={`${cls} bottom-2 left-2 border-b-2 border-l-2`} />
      <span className={`${cls} bottom-2 right-2 border-b-2 border-r-2`} />
    </>
  );
}

function BlueprintDiagram({ form }: { form: PropertyFeatures }) {
  return (
    <svg viewBox="0 0 400 420" className="w-full max-w-md" fill="none">
      <rect x="60" y="60" width="280" height="240" rx="2" stroke="var(--color-blueprint-line)" strokeWidth="1.5" opacity="0.6" />
      <line x1="200" y1="60" x2="200" y2="180" stroke="var(--color-blueprint-line)" strokeWidth="1" opacity="0.35" />
      <line x1="60" y1="180" x2="340" y2="180" stroke="var(--color-blueprint-line)" strokeWidth="1" opacity="0.35" />

      <line x1="60" y1="330" x2="340" y2="330" stroke="var(--color-gold)" strokeWidth="1" />
      <line x1="60" y1="322" x2="60" y2="338" stroke="var(--color-gold)" strokeWidth="1" />
      <line x1="340" y1="322" x2="340" y2="338" stroke="var(--color-gold)" strokeWidth="1" />
      <text x="200" y="352" textAnchor="middle" fill="var(--color-gold)" fontSize="11" fontFamily="var(--font-data)">
        {form.area_in_sqft.toLocaleString()} SQFT
      </text>

      <line x1="30" y1="60" x2="30" y2="300" stroke="var(--color-gold)" strokeWidth="1" />
      <line x1="22" y1="60" x2="38" y2="60" stroke="var(--color-gold)" strokeWidth="1" />
      <line x1="22" y1="300" x2="38" y2="300" stroke="var(--color-gold)" strokeWidth="1" />

      <text x="130" y="130" fill="var(--color-blueprint-line)" fontSize="11" fontFamily="var(--font-data)" opacity="0.8">
        {form.beds} BED
      </text>
      <text x="260" y="130" fill="var(--color-blueprint-line)" fontSize="11" fontFamily="var(--font-data)" opacity="0.8">
        {form.baths} BATH
      </text>
      <text x="130" y="245" fill="var(--color-blueprint-line)" fontSize="11" fontFamily="var(--font-data)" opacity="0.8">
        {form.type.toUpperCase()}
      </text>
      <text x="260" y="245" fill="var(--color-blueprint-line)" fontSize="11" fontFamily="var(--font-data)" opacity="0.8">
        {form.city.toUpperCase()}
      </text>

      {[[60, 60], [340, 60], [60, 300], [340, 300]].map(([x, y], i) => (
        <g key={i}>
          <line x1={x - 8} y1={y} x2={x + 8} y2={y} stroke="var(--color-gold)" strokeWidth="1" />
          <line x1={x} y1={y - 8} x2={x} y2={y + 8} stroke="var(--color-gold)" strokeWidth="1" />
        </g>
      ))}
    </svg>
  );
}