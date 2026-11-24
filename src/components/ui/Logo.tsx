import { cn } from '../../lib/utils';

interface LogoProps {
  collapsed?: boolean;
  className?: string;
}

export function Logo({ collapsed = false, className }: LogoProps) {
  return (
    <div className={cn('flex items-center gap-2.5 min-w-0', className)}>
      <div
        className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-gradient-to-br from-indigo-500 to-violet-600 text-white shadow-sm ring-1 ring-inset ring-white/15"
        aria-hidden="true"
      >
        <svg
          width="18"
          height="18"
          viewBox="0 0 24 24"
          fill="none"
          xmlns="http://www.w3.org/2000/svg"
        >
          <path
            d="M6.5 3.5H13.8L18.5 8.2V19.5C18.5 20.0523 18.0523 20.5 17.5 20.5H6.5C5.94772 20.5 5.5 20.0523 5.5 19.5V4.5C5.5 3.94772 5.94772 3.5 6.5 3.5Z"
            stroke="currentColor"
            strokeWidth="1.75"
            strokeLinejoin="round"
          />
          <path
            d="M13.5 3.5V8.5H18.5"
            stroke="currentColor"
            strokeWidth="1.75"
            strokeLinejoin="round"
          />
          <path
            d="M8.75 13H15.25M8.75 16.25H13"
            stroke="currentColor"
            strokeWidth="1.75"
            strokeLinecap="round"
          />
        </svg>
      </div>
      {!collapsed && (
        <span className="font-display text-[15px] font-semibold tracking-tight text-slate-900 dark:text-slate-50 whitespace-nowrap truncate">
          DocuMind AI
        </span>
      )}
    </div>
  );
}
