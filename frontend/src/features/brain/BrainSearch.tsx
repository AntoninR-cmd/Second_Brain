import { useEffect, useId, useState } from "react";

import type { BrainSearchResult } from "../../api/types";

interface BrainSearchProps {
  value: string;
  results: BrainSearchResult[];
  loading: boolean;
  error: string | null;
  onChange: (value: string) => void;
  onSelect: (result: BrainSearchResult) => void;
}

export function BrainSearch({
  value,
  results,
  loading,
  error,
  onChange,
  onSelect,
}: BrainSearchProps) {
  const listboxId = useId();
  const [activeIndex, setActiveIndex] = useState(-1);
  const open = value.trim().length >= 2;

  useEffect(() => {
    setActiveIndex(results.length > 0 ? 0 : -1);
  }, [results]);

  return (
    <div className="brain-search">
      <label className="sr-only" htmlFor={`${listboxId}-input`}>
        Rechercher un thème ou une connaissance
      </label>
      <span className="brain-search-icon" aria-hidden="true">
        ⌕
      </span>
      <input
        id={`${listboxId}-input`}
        type="search"
        value={value}
        placeholder="Rechercher un thème ou une connaissance…"
        autoComplete="off"
        aria-autocomplete="list"
        aria-controls={open ? listboxId : undefined}
        aria-expanded={open}
        aria-activedescendant={
          activeIndex >= 0 ? `${listboxId}-${activeIndex}` : undefined
        }
        onChange={(event) => onChange(event.target.value)}
        onKeyDown={(event) => {
          if (!open) return;
          if (event.key === "ArrowDown") {
            event.preventDefault();
            setActiveIndex((current) =>
              Math.min(results.length - 1, current + 1),
            );
          } else if (event.key === "ArrowUp") {
            event.preventDefault();
            setActiveIndex((current) => Math.max(0, current - 1));
          } else if (event.key === "Enter" && activeIndex >= 0) {
            const result = results[activeIndex];
            if (result) {
              event.preventDefault();
              onSelect(result);
            }
          } else if (event.key === "Escape") {
            onChange("");
          }
        }}
      />

      {loading ? (
        <span className="brain-search-loading" role="status">
          Recherche…
        </span>
      ) : null}

      {open ? (
        <div className="brain-search-results">
          {error ? <p role="alert">{error}</p> : null}
          {!loading && !error && results.length === 0 ? (
            <p>Aucun résultat dans ce cerveau.</p>
          ) : null}
          {results.length > 0 ? (
            <ul id={listboxId} role="listbox">
              {results.map((result, index) => (
                <li
                  id={`${listboxId}-${index}`}
                  key={`${result.kind}-${result.target_id}`}
                  role="option"
                  aria-selected={index === activeIndex}
                >
                  <button
                    type="button"
                    onMouseEnter={() => setActiveIndex(index)}
                    onClick={() => onSelect(result)}
                  >
                    <span className="brain-result-kind" aria-hidden="true">
                      {result.kind === "cluster" ? "◉" : "●"}
                    </span>
                    <span>
                      <strong>{result.label}</strong>
                      <small>
                        {result.kind === "cluster"
                          ? "Thème"
                          : result.source_title ?? "Connaissance"}
                      </small>
                    </span>
                  </button>
                </li>
              ))}
            </ul>
          ) : null}
        </div>
      ) : null}
    </div>
  );
}
