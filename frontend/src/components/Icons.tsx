type IconProps = { className?: string; "aria-hidden"?: boolean };

export function IconServer({ className, ...rest }: IconProps) {
  return (
    <svg className={className} width="20" height="20" viewBox="0 0 24 24" fill="none" {...rest}>
      <path
        d="M4 6h16v4H4V6zm0 8h16v4H4v-4z"
        stroke="currentColor"
        strokeWidth="1.75"
        strokeLinecap="round"
      />
      <circle cx="8" cy="8" r="1" fill="currentColor" />
      <circle cx="8" cy="16" r="1" fill="currentColor" />
    </svg>
  );
}

export function IconFlask({ className, ...rest }: IconProps) {
  return (
    <svg className={className} width="20" height="20" viewBox="0 0 24 24" fill="none" {...rest}>
      <path
        d="M10 3h4v5.5l4.5 9.5a2 2 0 0 1-1.8 2.9H7.3a2 2 0 0 1-1.8-2.9L10 8.5V3z"
        stroke="currentColor"
        strokeWidth="1.75"
        strokeLinejoin="round"
      />
      <path d="M9 14h6" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
    </svg>
  );
}

export function IconTable({ className, ...rest }: IconProps) {
  return (
    <svg className={className} width="20" height="20" viewBox="0 0 24 24" fill="none" {...rest}>
      <rect x="3" y="4" width="18" height="16" rx="2" stroke="currentColor" strokeWidth="1.75" />
      <path d="M3 10h18M10 10v10" stroke="currentColor" strokeWidth="1.75" />
    </svg>
  );
}

export function IconAlert({ className, ...rest }: IconProps) {
  return (
    <svg className={className} width="20" height="20" viewBox="0 0 24 24" fill="none" {...rest}>
      <path
        d="M12 3 22 19H2L12 3zM12 9v5M12 17h.01"
        stroke="currentColor"
        strokeWidth="1.75"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}
