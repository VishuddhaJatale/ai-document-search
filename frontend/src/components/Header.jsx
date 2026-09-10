function Header() {
  return (
    <header className="border-b border-slate-800/80 bg-[#080D1A]/95 backdrop-blur">

      <div className="mx-auto flex max-w-7xl items-center justify-between px-6 py-4">

        <div className="flex items-center gap-3">

          {/* Logo */}
          <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-gradient-to-br from-blue-500 to-violet-500 text-lg shadow-lg shadow-blue-500/20">
            ✦
          </div>

          <div>
            <h1 className="text-lg font-bold tracking-tight text-white">
              AI Document Intelligence
            </h1>

            <p className="text-xs text-slate-500">
              Intelligent document search & analysis
            </p>
          </div>

        </div>


        {/* Status */}
        <div className="hidden items-center gap-2 rounded-full border border-emerald-500/20 bg-emerald-500/5 px-3 py-1.5 sm:flex">

          <span className="h-1.5 w-1.5 rounded-full bg-emerald-400 shadow-sm shadow-emerald-400"></span>

          <span className="text-xs font-medium text-emerald-400">
            AI Ready
          </span>

        </div>

      </div>

    </header>
  );
}

export default Header;