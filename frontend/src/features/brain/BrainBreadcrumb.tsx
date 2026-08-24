export interface BrainBreadcrumbItem {
  id: string;
  label: string;
  level: number;
}

interface BrainBreadcrumbProps {
  items: BrainBreadcrumbItem[];
  onNavigate: (index: number) => void;
}

export function BrainBreadcrumb({ items, onNavigate }: BrainBreadcrumbProps) {
  return (
    <nav className="brain-breadcrumb" aria-label="Position dans le cerveau">
      <ol>
        {items.map((item, index) => {
          const current = index === items.length - 1;
          return (
            <li key={item.id}>
              {index > 0 ? <span aria-hidden="true">›</span> : null}
              <button
                type="button"
                aria-current={current ? "page" : undefined}
                disabled={current}
                onClick={() => onNavigate(index)}
              >
                {item.label}
              </button>
            </li>
          );
        })}
      </ol>
    </nav>
  );
}
