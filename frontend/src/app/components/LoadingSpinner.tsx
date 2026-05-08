export default function LoadingSpinner() {
  return (
    <div className="flex flex-col items-center justify-center h-full w-full gap-8">
      {/* Spinner */}
      <div className="relative w-32 h-32">
        <div className="absolute inset-0 border-8 border-white/20 rounded-full"></div>
        <div
          className="absolute inset-0 border-8 border-white border-t-transparent rounded-full animate-spin"
          style={{ animationDuration: "1s" }}
        ></div>
      </div>

      {/* Loading Text */}
      <div
        style={{ fontVariationSettings: "'GRAD' 0, 'wdth' 100" }}
        className="text-[36px] font-['Roboto_Serif:Medium',sans-serif] font-medium text-white text-center"
      >
        <p className="leading-[normal]">Analyzing review...</p>
      </div>

      {/* Animated Dots */}
      <div className="flex gap-3">
        <div
          className="w-4 h-4 bg-white rounded-full animate-bounce"
          style={{ animationDelay: "0ms", animationDuration: "1.4s" }}
        ></div>
        <div
          className="w-4 h-4 bg-white rounded-full animate-bounce"
          style={{ animationDelay: "200ms", animationDuration: "1.4s" }}
        ></div>
        <div
          className="w-4 h-4 bg-white rounded-full animate-bounce"
          style={{ animationDelay: "400ms", animationDuration: "1.4s" }}
        ></div>
      </div>
    </div>
  );
}
