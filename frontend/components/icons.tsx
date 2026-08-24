import type { SVGProps } from "react";

type IconProps = SVGProps<SVGSVGElement>;

export function Mark({ className, ...props }: IconProps) {
  return (
    <svg className={className} viewBox="0 0 44 44" fill="none" aria-hidden="true" {...props}>
      <path d="M5 18.8 22 9l17 9.8v16.4L22 44 5 35.2V18.8Z" fill="currentColor" opacity=".17" />
      <path d="M5 18.8 22 9l17 9.8L22 28.6 5 18.8Z" fill="currentColor" />
      <path d="M22 28.6v15.1M5 18.8v16.4l17 8.8 17-8.8V18.8" stroke="currentColor" strokeWidth="2" />
      <circle cx="22" cy="18.8" r="4.2" fill="#0B0D0D" />
    </svg>
  );
}

export function ArrowUpRight(props: IconProps) {
  return <svg viewBox="0 0 20 20" fill="none" aria-hidden="true" {...props}><path d="M5 15 15 5M7 5h8v8" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" /></svg>;
}

export function ArrowLeft(props: IconProps) {
  return <svg viewBox="0 0 20 20" fill="none" aria-hidden="true" {...props}><path d="M16 10H4m5-5-5 5 5 5" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" /></svg>;
}

export function Spark(props: IconProps) {
  return <svg viewBox="0 0 20 20" fill="none" aria-hidden="true" {...props}><path d="m10 1 1.8 6.2L18 9l-6.2 1.8L10 17l-1.8-6.2L2 9l6.2-1.8L10 1Z" stroke="currentColor" strokeWidth="1.5" strokeLinejoin="round" /><path d="m16.5 13 .7 2.3 2.3.7-2.3.7-.7 2.3-.7-2.3-2.3-.7 2.3-.7.7-2.3Z" fill="currentColor" /></svg>;
}

export function Pulse(props: IconProps) {
  return <svg viewBox="0 0 20 20" fill="none" aria-hidden="true" {...props}><path d="M1.5 10h3l1.7-5 3.1 10 2.2-7 1.5 2H18.5" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" /></svg>;
}

export function SettingsGlyph(props: IconProps) {
  return <svg viewBox="0 0 20 20" fill="none" aria-hidden="true" {...props}><path d="M10 7.2a2.8 2.8 0 1 0 0 5.6 2.8 2.8 0 0 0 0-5.6Zm7.6 3.9v-2.2l-2-.5a6.4 6.4 0 0 0-.6-1.5l1.1-1.8-1.6-1.6-1.8 1.1a6.4 6.4 0 0 0-1.5-.6l-.5-2h-2.2l-.5 2a6.4 6.4 0 0 0-1.5.6L4.8 3.5 3.2 5.1l1.1 1.8a6.4 6.4 0 0 0-.6 1.5l-2 .5v2.2l2 .5c.1.5.3 1 .6 1.5l-1.1 1.8 1.6 1.6 1.8-1.1c.5.3 1 .5 1.5.6l.5 2h2.2l.5-2c.5-.1 1-.3 1.5-.6l1.8 1.1 1.6-1.6-1.1-1.8c.3-.5.5-1 .6-1.5l2-.5Z" stroke="currentColor" strokeWidth="1.2" strokeLinejoin="round" /></svg>;
}

export function Check(props: IconProps) {
  return <svg viewBox="0 0 20 20" fill="none" aria-hidden="true" {...props}><path d="m4 10.5 3.8 3.8L16 6.2" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" /></svg>;
}

export function ListIcon(props: IconProps) {
  return <svg viewBox="0 0 20 20" fill="none" aria-hidden="true" {...props}><path d="M3 5h14M3 10h14M3 15h14" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" /></svg>;
}

export function PipelineIcon(props: IconProps) {
  return <svg viewBox="0 0 20 20" fill="none" aria-hidden="true" {...props}><path d="M2 10h4l2-4 3 8 2-6 2 2h3" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round" /><circle cx="5" cy="10" r="1.5" fill="currentColor" /><circle cx="15" cy="10" r="1.5" fill="currentColor" /></svg>;
}

export function ChartIcon(props: IconProps) {
  return <svg viewBox="0 0 20 20" fill="none" aria-hidden="true" {...props}><path d="M3 17V9m4.5 8V6M11 17V3m4.5 14V7" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" /></svg>;
}
