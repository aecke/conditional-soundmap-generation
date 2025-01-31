import json
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def plot_validation_trend():
    # Log-Datei laden
    log_path = "E:/Schallsimulationsdaten/urban_sound_25k_reflection/newtrainingsparameterCheckpoints/soundmap/512x512/glow_improved/building2soundmap/logs/training_log_20241217_011452.json"
    
    iterations = []
    val_losses = []
    
    with open(log_path, 'r') as f:
        for line in f:
            try:
                data = json.loads(line)
                if 'metrics' in data and 'val_loss' in data['metrics']:
                    if data['metrics']['val_loss'] is not None:
                        iterations.append(data['iteration'])
                        val_losses.append(data['metrics']['val_loss'])
            except:
                continue
    
    # Konvertiere zu numpy arrays
    iterations = np.array(iterations)
    val_losses = np.array(val_losses)
    
    # Berechne Trendlinie
    slope, intercept, r_value, p_value, std_err = stats.linregress(iterations, val_losses)
    trend_line = slope * iterations + intercept
    
    # Plotte Validation Loss und Trendlinie
    plt.figure(figsize=(12, 6))
    plt.plot(iterations, val_losses, 'b.', alpha=0.5, label='Validation Loss')
    plt.plot(iterations, trend_line, 'r-', label=f'Trend (slope = {slope:.2e})')
    
    plt.xlabel('Iteration')
    plt.ylabel('Validation Loss')
    plt.title('Validation Loss Over Training with Trend')
    plt.legend()
    plt.grid(True)
    
    # Füge R-squared Wert hinzu
    plt.text(0.02, 0.98, f'R² = {r_value**2:.4f}', 
             transform=plt.gca().transAxes, 
             bbox=dict(facecolor='white', alpha=0.8))
    
    plt.show()
    
    # Drucke Statistiken
    print(f"Anzahl Validierungspunkte: {len(val_losses)}")
    print(f"Durchschnittlicher Validation Loss: {np.mean(val_losses):.4f}")
    print(f"Minimum Validation Loss: {np.min(val_losses):.4f}")
    print(f"Maximum Validation Loss: {np.max(val_losses):.4f}")
    
plot_validation_trend()