export default function Spinner({ size = 20, label }) {
  return (
    <div className="flex items-center gap-2 text-ink-700">
      <span
        className="inline-block rounded-full border-2 border-beige-300 border-t-ink-900 animate-spin"
        style={{ width: size, height: size }}
      />
      {label && <span className="text-sm">{label}</span>}
    </div>
  )
}
