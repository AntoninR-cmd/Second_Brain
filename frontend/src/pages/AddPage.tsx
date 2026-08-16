import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useState, type FormEvent } from "react";
import { Link, useNavigate } from "react-router-dom";

import {
  createManualSource,
  getReadableError,
} from "../api/client";
import type { ManualSourceInput } from "../api/types";

export function AddPage() {
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const [title, setTitle] = useState("");
  const [author, setAuthor] = useState("");
  const [text, setText] = useState("");

  const createNote = useMutation({
    mutationFn: createManualSource,
    onSuccess: async (source) => {
      await queryClient.invalidateQueries({ queryKey: ["dashboard"] });
      navigate("/", {
        replace: true,
        state: { flash: `La note « ${source.title} » a bien été enregistrée.` },
      });
    },
  });

  const trimmedText = text.trim();
  const canSubmit = trimmedText.length > 0 && !createNote.isPending;

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();

    if (!trimmedText) {
      return;
    }

    const trimmedTitle = title.trim();
    const trimmedAuthor = author.trim();
    const input: ManualSourceInput = {
      text,
      ...(trimmedTitle ? { title: trimmedTitle } : {}),
      ...(trimmedAuthor ? { author: trimmedAuthor } : {}),
    };

    createNote.mutate(input);
  }

  function clearMutationError() {
    if (createNote.isError) {
      createNote.reset();
    }
  }

  return (
    <section className="page narrow-page">
      <header className="page-header">
        <div>
          <p className="eyebrow">Nouvelle entrée</p>
          <h1>Ajouter une note</h1>
          <p className="page-introduction">
            Collez ou saisissez un texte. Il sera conservé tel quel dans votre base
            locale.
          </p>
        </div>
      </header>

      <form className="panel note-form" onSubmit={handleSubmit}>
        <div className="form-section">
          <div className="field-group">
            <label htmlFor="note-title">
              Titre <span className="optional-label">Facultatif</span>
            </label>
            <input
              id="note-title"
              name="title"
              type="text"
              maxLength={255}
              autoComplete="off"
              placeholder="Ex. Idées pour mon prochain projet"
              value={title}
              onChange={(event) => {
                setTitle(event.target.value);
                clearMutationError();
              }}
            />
            <p className="field-help">
              Sans titre, la première ligne du texte sera utilisée.
            </p>
          </div>

          <div className="field-group">
            <label htmlFor="note-author">
              Auteur <span className="optional-label">Facultatif</span>
            </label>
            <input
              id="note-author"
              name="author"
              type="text"
              maxLength={255}
              autoComplete="off"
              placeholder="Ex. Antonin"
              value={author}
              onChange={(event) => {
                setAuthor(event.target.value);
                clearMutationError();
              }}
            />
          </div>

          <div className="field-group">
            <div className="label-row">
              <label htmlFor="note-text">Texte</label>
              <span className="character-count" aria-live="polite">
                {text.length.toLocaleString("fr-FR")} caractère
                {text.length > 1 ? "s" : ""}
              </span>
            </div>
            <textarea
              id="note-text"
              name="text"
              rows={14}
              required
              aria-required="true"
              aria-describedby="text-help"
              placeholder="Écrivez votre note ici…"
              value={text}
              onChange={(event) => {
                setText(event.target.value);
                clearMutationError();
              }}
            />
            <p id="text-help" className="field-help">
              Le texte est obligatoire. Les espaces seuls ne sont pas enregistrés.
            </p>
          </div>
        </div>

        {createNote.isError ? (
          <div className="alert alert-error form-alert" role="alert">
            <span aria-hidden="true">!</span>
            <p>{getReadableError(createNote.error)}</p>
          </div>
        ) : null}

        <div className="form-actions">
          <Link className="button button-ghost" to="/">
            Annuler
          </Link>
          <button
            className="button button-primary"
            type="submit"
            disabled={!canSubmit}
          >
            {createNote.isPending ? (
              <>
                <span className="spinner spinner-light" aria-hidden="true" />
                Enregistrement…
              </>
            ) : (
              "Enregistrer la note"
            )}
          </button>
        </div>
      </form>
    </section>
  );
}
