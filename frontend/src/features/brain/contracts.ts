/**
 * Feature-local aliases of the FastAPI contracts.
 *
 * Keeping the aliases here makes the graph modules independent from fetching
 * while retaining a single source of truth for the HTTP response shapes.
 */
import type {
  BrainCluster,
  BrainClusterDetail,
  BrainGraphEdge,
  BrainGraphNode,
  BrainGraphResponse,
  BrainKnowledgeNode,
} from "../../api/types";

export type BrainGraphNodeKind = BrainGraphNode["kind"];
export type BrainGraphApiNode = BrainGraphNode;
export type BrainGraphApiEdge = BrainGraphEdge;
export type BrainGraphApiResponse = BrainGraphResponse;
export type BrainClusterApiItem = BrainCluster;
export type BrainClusterApiDetail = BrainClusterDetail;
export type BrainKnowledgeApiItem = BrainKnowledgeNode;
