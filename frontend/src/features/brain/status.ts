import type { BrainStatus } from "../../api/types";

export interface BrainAvailability {
  canDisplayGraph: boolean;
  tone: "ready" | "info" | "warning" | "error";
  title: string;
  message: string | null;
}

export function getBrainAvailability(status: BrainStatus): BrainAvailability {
  if (status.active_profile !== null) {
    if (status.state === "stale") {
      return {
        canDisplayGraph: true,
        tone: "warning",
        title: "Cerveau disponible, mais incomplet",
        message: "Le cerveau ne contient pas encore les dernières connaissances.",
      };
    }
    if (status.state === "building") {
      return {
        canDisplayGraph: true,
        tone: "info",
        title: "Reconstruction en cours",
        message: "La dernière version valide reste utilisable pendant la reconstruction.",
      };
    }
    if (status.state === "error") {
      return {
        canDisplayGraph: true,
        tone: "warning",
        title: "La dernière reconstruction a échoué",
        message: status.error ?? "La dernière version valide reste consultable.",
      };
    }
    return {
      canDisplayGraph: true,
      tone: "ready",
      title: "Cerveau prêt",
      message: null,
    };
  }

  switch (status.state) {
    case "building":
      return {
        canDisplayGraph: false,
        tone: "info",
        title: "Construction du cerveau en cours",
        message: status.active_job?.progress_message ?? "La carte sera disponible à la fin du traitement.",
      };
    case "vector_index_required":
      return {
        canDisplayGraph: false,
        tone: "warning",
        title: "Index vectoriel requis",
        message: "Indexez d’abord les connaissances dans les Paramètres.",
      };
    case "error":
    case "unavailable":
      return {
        canDisplayGraph: false,
        tone: "error",
        title: "Cerveau indisponible",
        message: status.error ?? "Consultez les Paramètres pour corriger le problème.",
      };
    case "empty":
    case "not_built":
    case "ready":
    case "stale":
      return {
        canDisplayGraph: false,
        tone: "info",
        title: "Aucun cerveau construit",
        message: "Construisez la carte depuis les Paramètres après avoir indexé vos connaissances.",
      };
  }
}
