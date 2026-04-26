import { NavLink, Outlet } from 'react-router-dom'
import { Toaster } from 'react-hot-toast'
import { FiHome, FiActivity, FiMessageSquare, FiBookOpen } from 'react-icons/fi'

export default function Layout() {
  const navItems = [
    { to: '/', label: 'Home', icon: FiHome, end: true },
    { to: '/predict', label: 'Predict', icon: FiActivity },
    { to: '/chat', label: 'Chat', icon: FiMessageSquare },
    { to: '/explain', label: 'Explain', icon: FiBookOpen },
  ]

  return (
    <div className="min-h-screen bg-beige-50">
      <Toaster
        position="top-right"
        toastOptions={{
          style: { background: '#1a1a1a', color: '#f5efe2' },
        }}
      />

      {/* Header */}
      <header className="bg-ink-900 text-beige-50 sticky top-0 z-30 shadow-soft">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex items-center justify-between h-16">
          <NavLink to="/" className="flex items-center gap-2">
            <div className="w-9 h-9 rounded-lg bg-beige-200 flex items-center justify-center">
              <span className="text-ink-900 font-serif font-bold text-lg">M</span>
            </div>
            <div className="leading-tight">
              <div className="font-serif font-bold text-lg tracking-wide">MedCareAI</div>
              <div className="text-[10px] uppercase tracking-widest text-beige-300">
                Decision Support
              </div>
            </div>
          </NavLink>

          <nav className="hidden md:flex items-center gap-1">
            {navItems.map(({ to, label, icon: Icon, end }) => (
              <NavLink
                key={to}
                to={to}
                end={end}
                className={({ isActive }) =>
                  `flex items-center gap-2 px-4 py-2 rounded-md text-sm font-medium transition-colors ${
                    isActive
                      ? 'bg-beige-200 text-ink-900'
                      : 'text-beige-200 hover:bg-ink-800 hover:text-beige-50'
                  }`
                }
              >
                <Icon size={16} />
                {label}
              </NavLink>
            ))}
          </nav>
        </div>

        {/* Mobile nav */}
        <nav className="md:hidden flex justify-around border-t border-ink-800 py-2">
          {navItems.map(({ to, label, icon: Icon, end }) => (
            <NavLink
              key={to}
              to={to}
              end={end}
              className={({ isActive }) =>
                `flex flex-col items-center gap-0.5 text-xs px-2 py-1 ${
                  isActive ? 'text-beige-200' : 'text-beige-400'
                }`
              }
            >
              <Icon size={18} />
              {label}
            </NavLink>
          ))}
        </nav>
      </header>

      {/* Main content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <Outlet />
      </main>

      {/* Footer */}
      <footer className="border-t border-beige-200 mt-16 py-6 bg-beige-100">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center text-sm text-ink-600">
          <p className="font-serif italic">
            MedCareAI &mdash; AI-powered medical decision support.
          </p>
          <p className="text-xs mt-1 text-ink-500">
            Educational tool only. Not a substitute for professional medical advice.
          </p>
        </div>
      </footer>
    </div>
  )
}
