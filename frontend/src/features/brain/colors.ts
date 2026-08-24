import type { BrainGraphNodeKind } from "./contracts";

/** Stable FNV-1a hash: no dependency on labels or their language. */
export function stableBrainHash(value: string): number {
  let hash = 0x811c9dc5;
  for (let index = 0; index < value.length; index += 1) {
    hash ^= value.charCodeAt(index);
    hash = Math.imul(hash, 0x01000193);
  }
  return hash >>> 0;
}

export function brainFamilyHue(familyId: string): number {
  return stableBrainHash(familyId) % 360;
}

export function colorForBrainNode(
  familyId: string,
  variantId: string,
  kind: BrainGraphNodeKind,
): string {
  if (familyId === "unassigned") {
    return kind === "cluster" ? "#94a3b8" : "#aab5c5";
  }

  const baseHue = brainFamilyHue(familyId);
  // A child keeps its domain hue while gaining a small deterministic variation.
  const variantOffset = (stableBrainHash(variantId) % 19) - 9;
  const hue = (baseHue + variantOffset + 360) % 360;
  const saturation = kind === "cluster" ? 64 : 58;
  const lightness = kind === "cluster" ? 59 : 68;
  return hslToHex(hue, saturation, lightness);
}

export function visualNodeSize(kind: BrainGraphNodeKind, rawSize: number): number {
  if (kind === "knowledge") {
    return 5.5;
  }
  const safeSize = Number.isFinite(rawSize) ? Math.max(1, rawSize) : 1;
  return clamp(8 + Math.sqrt(safeSize) * 1.85, 10, 26);
}

export function visualEdgeSize(relationCount: number): number {
  const safeCount = Number.isFinite(relationCount)
    ? Math.max(1, relationCount)
    : 1;
  return clamp(0.4 + Math.log2(safeCount + 1) * 0.28, 0.68, 2.1);
}

export function visualEdgeColor(score: number, emphasized = false): string {
  const safeScore = clamp(Number.isFinite(score) ? score : 0, 0, 1);
  const alpha = emphasized ? 0.72 : 0.07 + safeScore * 0.17;
  return `rgba(148, 163, 184, ${alpha.toFixed(3)})`;
}

export function dimmedNodeColor(): string {
  return "rgba(91, 103, 126, 0.24)";
}

function hslToHex(hue: number, saturation: number, lightness: number): string {
  const s = saturation / 100;
  const l = lightness / 100;
  const chroma = (1 - Math.abs(2 * l - 1)) * s;
  const hueSection = hue / 60;
  const second = chroma * (1 - Math.abs((hueSection % 2) - 1));
  let red = 0;
  let green = 0;
  let blue = 0;

  if (hueSection < 1) {
    red = chroma;
    green = second;
  } else if (hueSection < 2) {
    red = second;
    green = chroma;
  } else if (hueSection < 3) {
    green = chroma;
    blue = second;
  } else if (hueSection < 4) {
    green = second;
    blue = chroma;
  } else if (hueSection < 5) {
    red = second;
    blue = chroma;
  } else {
    red = chroma;
    blue = second;
  }

  const match = l - chroma / 2;
  return `#${[red, green, blue]
    .map((channel) => Math.round((channel + match) * 255))
    .map((channel) => channel.toString(16).padStart(2, "0"))
    .join("")}`;
}

function clamp(value: number, minimum: number, maximum: number): number {
  return Math.min(maximum, Math.max(minimum, value));
}
