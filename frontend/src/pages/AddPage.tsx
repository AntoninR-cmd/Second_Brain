import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useState, type ChangeEvent, type FormEvent } from "react";
import { Link, useNavigate } from "react-router-dom";

import {
  createManualSource,
  getReadableError,
  uploadSource,
} from "../api/client";
import type { FileSourceInput, ManualSourceInput } from "../api/types";

type AddMode = "manual" | "file";

function ManualSourceForm() {
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const [title, setTitle] = useState("");
  const [author, setAuthor] = useState("");
  const [text, setText] = useState("");

  const createNote = useMutation({
    mutationFn: createManualSource,
    onSuccess: async (source) => {
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: ["dashboard"] }),
        queryClient.invalidateQueries({ queryKey: ["sources"] }),
      ]);
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
  );
}

function filenameTitle(filename: string): string {
  const withoutExtension = filename.replace(/\.[^.]+$/, "").trim();
  return withoutExtension || "Source importée";
}

function formatFileSize(size: number): string {
  if (size < 1_024) {
    return `${size} octet${size > 1 ? "s" : ""}`;
  }

  if (size < 1_048_576) {
    return `${(size / 1_024).toLocaleString("fr-FR", {
      maximumFractionDigits: 1,
    })} Ko`;
  }

  return `${(size / 1_048_576).toLocaleString("fr-FR", {
    maximumFractionDigits: 1,
  })} Mo`;
}

function FileSourceForm() {
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const [file, setFile] = useState<File | null>(null);
  const [title, setTitle] = useState("");
  const [titleWasEdited, setTitleWasEdited] = useState(false);
  const [author, setAuthor] = useState("");
  const [fileError, setFileError] = useState<string | null>(null);

  const importSource = useMutation({
    mutationFn: uploadSource,
    onSuccess: async (source) => {
      queryClient.setQueryData(["sources", source.id], source);
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: ["dashboard"] }),
        queryClient.invalidateQueries({ queryKey: ["sources"] }),
      ]);
      navigate(`/sources/${source.id}`, {
        replace: true,
        state: {
          flash: `La source « ${source.title} » a bien été importée.`,
        },
      });
    },
  });

  function clearMutationError() {
    if (importSource.isError) {
      importSource.reset();
    }
  }

  function handleFileChange(event: ChangeEvent<HTMLInputElement>) {
    const selectedFile = event.target.files?.[0] ?? null;
    setFileError(null);
    clearMutationError();

    if (!selectedFile) {
      setFile(null);
      return;
    }

    const extension = selectedFile.name.split(".").pop()?.toLowerCase();
    if (extension !== "srt" && extension !== "txt") {
      setFile(null);
      setFileError("Sélectionnez uniquement un fichier .srt ou .txt.");
      event.target.value = "";
      return;
    }

    if (selectedFile.size === 0) {
      setFile(null);
      setFileError("Le fichier sélectionné est vide.");
      event.target.value = "";
      return;
    }

    setFile(selectedFile);
    if (!titleWasEdited) {
      setTitle(filenameTitle(selectedFile.name));
    }
  }

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();

    if (!file) {
      setFileError("Sélectionnez un fichier .srt ou .txt à importer.");
      return;
    }

    const trimmedTitle = title.trim();
    const trimmedAuthor = author.trim();
    const input: FileSourceInput = {
      file,
      ...(trimmedTitle ? { title: trimmedTitle } : {}),
      ...(trimmedAuthor ? { author: trimmedAuthor } : {}),
    };

    importSource.mutate(input);
  }

  return (
    <form className="panel note-form" onSubmit={handleSubmit}>
      <div className="form-section">
        <div className="field-group">
          <label htmlFor="source-file">Fichier SRT ou TXT</label>
          <label className="file-picker" htmlFor="source-file">
            <span className="file-picker-icon" aria-hidden="true">
              ↑
            </span>
            <span>
              <strong>Choisir un fichier</strong>
              <small>Formats acceptés : .srt et .txt</small>
            </span>
          </label>
          <input
            className="visually-hidden file-input"
            id="source-file"
            name="file"
            type="file"
            accept=".srt,.txt"
            required
            aria-required="true"
            aria-describedby="file-help"
            onChange={handleFileChange}
          />
          {file ? (
            <div className="selected-file" role="status">
              <span>
                <strong>{file.name}</strong>
                <small>{formatFileSize(file.size)}</small>
              </span>
              <span className="selected-file-type">
                {file.name.toLowerCase().endsWith(".srt") ? "SRT" : "TXT"}
              </span>
            </div>
          ) : null}
          <p id="file-help" className="field-help">
            Une copie intacte sera conservée dans le dossier de données local.
          </p>
          {fileError ? (
            <p className="field-error" role="alert">
              {fileError}
            </p>
          ) : null}
        </div>

        <div className="field-group">
          <label htmlFor="file-title">
            Titre <span className="optional-label">Facultatif</span>
          </label>
          <input
            id="file-title"
            name="title"
            type="text"
            maxLength={255}
            autoComplete="off"
            placeholder="Dérivé automatiquement du nom du fichier"
            value={title}
            onChange={(event) => {
              setTitle(event.target.value);
              setTitleWasEdited(true);
              clearMutationError();
            }}
          />
          <p className="field-help">
            Si vous le laissez vide, le nom du fichier sera utilisé.
          </p>
        </div>

        <div className="field-group">
          <label htmlFor="file-author">
            Auteur <span className="optional-label">Facultatif</span>
          </label>
          <input
            id="file-author"
            name="author"
            type="text"
            maxLength={255}
            autoComplete="off"
            placeholder="Ex. Nom de l’intervenant ou de l’auteur"
            value={author}
            onChange={(event) => {
              setAuthor(event.target.value);
              clearMutationError();
            }}
          />
        </div>
      </div>

      {importSource.isError ? (
        <div className="alert alert-error form-alert" role="alert">
          <span aria-hidden="true">!</span>
          <p>{getReadableError(importSource.error)}</p>
        </div>
      ) : null}

      <div className="form-actions">
        <Link className="button button-ghost" to="/sources">
          Annuler
        </Link>
        <button
          className="button button-primary"
          type="submit"
          disabled={!file || importSource.isPending}
        >
          {importSource.isPending ? (
            <>
              <span className="spinner spinner-light" aria-hidden="true" />
              Import en cours…
            </>
          ) : (
            "Importer le fichier"
          )}
        </button>
      </div>
    </form>
  );
}

export function AddPage() {
  const [mode, setMode] = useState<AddMode>("manual");

  return (
    <section className="page narrow-page">
      <header className="page-header">
        <div>
          <p className="eyebrow">Nouvelle entrée</p>
          <h1>Ajouter une source</h1>
          <p className="page-introduction">
            Saisissez une note libre ou importez un fichier de sous-titres SRT ou
            un document texte TXT.
          </p>
        </div>
      </header>

      <div className="add-mode-switch" role="group" aria-label="Type d’ajout">
        <button
          className={`add-mode-button${mode === "manual" ? " is-active" : ""}`}
          type="button"
          aria-pressed={mode === "manual"}
          onClick={() => setMode("manual")}
        >
          <span className="add-mode-icon" aria-hidden="true">
            ✎
          </span>
          <span>
            <strong>Ajout manuel</strong>
            <small>Écrire ou coller une note</small>
          </span>
        </button>
        <button
          className={`add-mode-button${mode === "file" ? " is-active" : ""}`}
          type="button"
          aria-pressed={mode === "file"}
          onClick={() => setMode("file")}
        >
          <span className="add-mode-icon" aria-hidden="true">
            ↑
          </span>
          <span>
            <strong>Importer un fichier</strong>
            <small>SRT ou TXT uniquement</small>
          </span>
        </button>
      </div>

      <div>
        {mode === "manual" ? <ManualSourceForm /> : <FileSourceForm />}
      </div>
    </section>
  );
}
