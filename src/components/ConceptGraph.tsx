import { GitBranch, Network, Target } from "lucide-react";
import { useMemo } from "react";
import {
  buildConceptNeighborhood,
  conceptDomainColors,
  type ConceptGraphNode,
} from "../data/conceptGraph";
import type { AlgorithmDefinition } from "../types/algorithm";

type Props = {
  algorithms: AlgorithmDefinition[];
  activeId: string;
  onSelect: (id: string) => void;
};

type PositionedNode = ConceptGraphNode & {
  x: number;
  y: number;
};

const width = 920;
const height = 330;
const graphCenter = { x: 310, y: 166 };
const orbit = { x: 238, y: 116 };

export function ConceptGraph({ algorithms, activeId, onSelect }: Props) {
  const neighborhood = useMemo(
    () => buildConceptNeighborhood(algorithms, activeId),
    [activeId, algorithms],
  );
  const nodes = useMemo(() => positionNodes(neighborhood.nodes), [neighborhood.nodes]);
  const nodeById = new Map(nodes.map((node) => [node.id, node]));
  const algorithmById = new Map(algorithms.map((algorithm) => [algorithm.id, algorithm]));
  const activeNode = nodeById.get(neighborhood.active.id);
  const neighborRows = neighborhood.directRelationships
    .map((relationship) => {
      const neighborId =
        relationship.source === neighborhood.active.id
          ? relationship.target
          : relationship.source;
      const algorithm = algorithmById.get(neighborId);
      return algorithm ? { algorithm, relationship } : null;
    })
    .filter((row): row is NonNullable<typeof row> => Boolean(row));

  const selectNode = (id: string) => {
    onSelect(id);
  };

  return (
    <section className="concept-graph-card" aria-label="Concept relationship graph">
      <div className="concept-graph-header">
        <div>
          <span className="eyebrow">Concept Map</span>
          <h3>Learning graph</h3>
        </div>
        <div className="graph-legend" aria-label="Concept domains">
          <LegendDot color={conceptDomainColors.supervised} label="supervised" />
          <LegendDot color={conceptDomainColors.neural} label="neural" />
          <LegendDot color={conceptDomainColors.optimization} label="optimization" />
        </div>
      </div>

      <div className="concept-graph-body">
        <svg
          className="concept-graph-svg"
          viewBox={`0 0 ${width} ${height}`}
          role="img"
          aria-labelledby="concept-graph-title"
        >
          <title id="concept-graph-title">
            {neighborhood.active.name} and connected algorithm lessons
          </title>
          <defs>
            <filter id="node-shadow" x="-25%" y="-25%" width="150%" height="150%">
              <feDropShadow dx="0" dy="8" stdDeviation="8" floodOpacity="0.14" />
            </filter>
          </defs>
          <g className="graph-links">
            {neighborhood.links.map((link) => {
              const source = nodeById.get(link.source);
              const target = nodeById.get(link.target);

              if (!source || !target) {
                return null;
              }

              const label = link.sourceDepth === 0 || link.targetDepth === 0 ? link.label : "";
              const midX = (source.x + target.x) / 2;
              const midY = (source.y + target.y) / 2;

              return (
                <g key={`${link.source}-${link.target}`}>
                  <line
                    x1={source.x}
                    y1={source.y}
                    x2={target.x}
                    y2={target.y}
                    className={
                      link.sourceDepth === 0 || link.targetDepth === 0
                        ? "graph-link primary"
                        : "graph-link"
                    }
                  />
                  {label && (
                    <text className="graph-link-label" x={midX} y={midY - 7}>
                      {truncate(label, 22)}
                    </text>
                  )}
                </g>
              );
            })}
          </g>

          <g className="graph-nodes">
            {nodes.map((node) => {
              const active = node.id === neighborhood.active.id;
              const color = conceptDomainColors[node.domain];
              const radius = active ? 36 : 24;

              return (
                <g
                  key={node.id}
                  className={`graph-node ${active ? "active" : ""}`}
                  role="button"
                  tabIndex={0}
                  aria-label={`${node.algorithm.name}, ${node.algorithm.category}`}
                  onClick={() => selectNode(node.id)}
                  onKeyDown={(event) => {
                    if (event.key === "Enter" || event.key === " ") {
                      event.preventDefault();
                      selectNode(node.id);
                    }
                  }}
                >
                  <circle
                    cx={node.x}
                    cy={node.y}
                    r={radius}
                    fill={active ? color : "#ffffff"}
                    stroke={color}
                    strokeWidth={active ? 4 : 2.5}
                    filter={active ? "url(#node-shadow)" : undefined}
                  />
                  <circle
                    cx={node.x}
                    cy={node.y}
                    r={active ? 7 : 5}
                    fill={active ? "#ffffff" : color}
                  />
                  <text
                    className={`graph-node-label ${active ? "active" : ""}`}
                    x={node.x}
                    y={node.y + radius + 18}
                  >
                    {truncate(node.algorithm.name, active ? 26 : 20)}
                  </text>
                </g>
              );
            })}
          </g>

          {activeNode && (
            <g className="graph-focus-ring" aria-hidden="true">
              <circle cx={activeNode.x} cy={activeNode.y} r="49" />
            </g>
          )}
        </svg>

        <aside className="concept-graph-detail" aria-label="Focused concept details">
          <div className="detail-kicker">
            <Target size={15} aria-hidden="true" />
            <span>{neighborhood.active.category}</span>
          </div>
          <h4>{neighborhood.active.name}</h4>
          <p>{neighborhood.active.summary}</p>

          <div className="concept-neighbor-list">
            <span className="neighbor-heading">
              <Network size={15} aria-hidden="true" />
              Connected lessons
            </span>
            {neighborRows.slice(0, 6).map(({ algorithm, relationship }) => (
              <button
                className="neighbor-row"
                key={`${relationship.source}-${relationship.target}`}
                type="button"
                onClick={() => selectNode(algorithm.id)}
              >
                <GitBranch size={14} aria-hidden="true" />
                <span>
                  <strong>{algorithm.name}</strong>
                  <small>{relationship.label}</small>
                </span>
              </button>
            ))}
          </div>
        </aside>
      </div>
    </section>
  );
}

function positionNodes(nodes: ConceptGraphNode[]): PositionedNode[] {
  const sorted = [...nodes].sort((a, b) => {
    if (a.depth !== b.depth) {
      return a.depth - b.depth;
    }

    return a.algorithm.name.localeCompare(b.algorithm.name);
  });
  const neighbors = sorted.filter((node) => node.depth === 1);
  const angleStart = -Math.PI / 2;
  const angleStep = neighbors.length > 1 ? (Math.PI * 2) / neighbors.length : 0;
  const neighborPositions = new Map(
    neighbors.map((node, index) => {
      const angle = angleStart + index * angleStep;
      return [
        node.id,
        {
          x: graphCenter.x + Math.cos(angle) * orbit.x,
          y: graphCenter.y + Math.sin(angle) * orbit.y,
        },
      ];
    }),
  );

  return sorted.map((node) => {
    if (node.depth === 0) {
      return {
        ...node,
        ...graphCenter,
      };
    }

    const position = neighborPositions.get(node.id) ?? graphCenter;
    return {
      ...node,
      ...position,
    };
  });
}

function LegendDot({ color, label }: { color: string; label: string }) {
  return (
    <span>
      <i style={{ backgroundColor: color }} aria-hidden="true" />
      {label}
    </span>
  );
}

function truncate(value: string, maxLength: number) {
  return value.length <= maxLength ? value : `${value.slice(0, maxLength - 3)}...`;
}
