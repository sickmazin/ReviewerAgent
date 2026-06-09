import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_benchmark_results(csv_path: str):
    if not os.path.exists(csv_path):
        print(f"[ERROR] Impossibile trovare il file: {csv_path}")
        return

    print(f"Caricamento dati da: {csv_path}")
    df = pd.read_csv(csv_path)

    # Assicurati che i dati numerici siano trattati correttamente, ignorando i None/NaN
    df = df.dropna(subset=['score_insight', 'score_gemini', 'score_ollama'])
    if df.empty:
        print("[ERROR] Nessun dato valido (confezionato da tutti e 3 i modelli) da plottare.")
        return

    print(f"Generazione grafici per {len(df)} recensioni valide...")

    # Stile generale
    sns.set_theme(style="whitegrid", palette="muted")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Insightfulness Model vs Cloud LLMs Benchmark", fontsize=18, fontweight='bold')

    # ---------------------------------------------------------
    # 1. Mean Average Time (Bar Plot)
    # ---------------------------------------------------------
    ax_time = axes[0, 0]
    avg_times = [df['time_insight'].mean(), df['time_gemini'].mean(), df['time_ollama'].mean()]
    models = ['Insightfulness (Local)', 'Gemini 3.1 Flash', 'Ollama Cloud']
    colors = ['#4c72b0', '#dd8452', '#55a868']
    
    bars = ax_time.bar(models, avg_times, color=colors)
    ax_time.set_title('Mean Execution Time per Review', fontsize=14)
    ax_time.set_ylabel('Time (seconds)')
    
    # Aggiungi i valori sopra le barre
    for bar in bars:
        yval = bar.get_height()
        ax_time.text(bar.get_x() + bar.get_width()/2, yval + 0.05, f"{yval:.2f}s", ha='center', va='bottom', fontweight='bold')

    # ---------------------------------------------------------
    # 2. Score Difference Distributions (Histograms)
    # ---------------------------------------------------------
    ax_diff = axes[0, 1]
    
    # Calcola le differenze
    diff_gemini = df['score_insight'] - df['score_gemini']
    diff_ollama = df['score_insight'] - df['score_ollama']
    
    sns.histplot(diff_gemini, kde=True, ax=ax_diff, color='#dd8452', label='vs Gemini', alpha=0.5)
    sns.histplot(diff_ollama, kde=True, ax=ax_diff, color='#55a868', label='vs Ollama', alpha=0.5)
    
    # Calcola deviazione standard e media della differenza
    std_gemini = diff_gemini.std()
    std_ollama = diff_ollama.std()
    
    ax_diff.set_title('Score Differences (Insightfulness - Cloud)', fontsize=14)
    ax_diff.set_xlabel('Score Difference')
    ax_diff.set_ylabel('Frequency')
    ax_diff.legend(title=f"Std Dev vs Gemini: {std_gemini:.2f}\nStd Dev vs Ollama: {std_ollama:.2f}")

    # ---------------------------------------------------------
    # 3. Correlation Scatter Plot
    # ---------------------------------------------------------
    ax_corr = axes[1, 0]
    
    sns.scatterplot(x='score_insight', y='score_gemini', data=df, ax=ax_corr, color='#dd8452', label='Gemini', s=60, alpha=0.7)
    sns.scatterplot(x='score_insight', y='score_ollama', data=df, ax=ax_corr, color='#55a868', label='Ollama', s=60, alpha=0.7, marker='X')
    
    # Linea perfetta concordanza
    lims = [
        np.min([ax_corr.get_xlim(), ax_corr.get_ylim()]),  
        np.max([ax_corr.get_xlim(), ax_corr.get_ylim()]),  
    ]
    ax_corr.plot(lims, lims, 'k-', alpha=0.3, zorder=0, label='Perfect Match')
    
    # Calcola correlazione di Pearson
    corr_gemini = df['score_insight'].corr(df['score_gemini'])
    corr_ollama = df['score_insight'].corr(df['score_ollama'])
    
    ax_corr.set_title('Score Correlation (Local vs Cloud)', fontsize=14)
    ax_corr.set_xlabel('Insightfulness (Local Model) Score')
    ax_corr.set_ylabel('Cloud LLM Score')
    ax_corr.legend(title=f"Pearson vs Gemini: {corr_gemini:.2f}\nPearson vs Ollama: {corr_ollama:.2f}")

    # ---------------------------------------------------------
    # 4. Boxplot of Scores (Distribution Spread)
    # ---------------------------------------------------------
    ax_box = axes[1, 1]
    
    score_data = pd.DataFrame({
        'Insightfulness': df['score_insight'],
        'Gemini': df['score_gemini'],
        'Ollama': df['score_ollama']
    })
    
    sns.boxplot(data=score_data, ax=ax_box, palette=colors)
    ax_box.set_title('Score Distributions Spread', fontsize=14)
    ax_box.set_ylabel('Assigned Score (0-100)')

    # Layout e salvataggio
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    output_img_path = os.path.join(os.path.dirname(csv_path), "benchmark_plots.png")
    plt.savefig(output_img_path, dpi=300)
    print(f"\nGrafici salvati con successo in: {output_img_path}")
    
    # Opzionale: mostra la finestra se in esecuzione interattiva
    # plt.show()

if __name__ == "__main__":
    csv_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "benchmark_results.csv"))
    plot_benchmark_results(csv_file)
