import pandas as pd
import os
import argparse
from tqdm import tqdm
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

def analyze_conditions(data_dir, pred_dir, output_dir, model_type, detailed_results_path):
    """
    Analysiert den Zusammenhang zwischen Bedingungen, Vorhersagen und Verlustwerten.
    Integriert pixelweise dB-Werte aus PNG-Bildern, visualisiert die besten/schlechtesten Predictions
    und hebt die Pixel mit den höchsten wMAPE-Fehlern hervor.

    Args:
        data_dir: Verzeichnis mit den Testdaten.
        pred_dir: Verzeichnis mit den Vorhersagen (PNG-Bilder).
        output_dir: Basisverzeichnis für die Auswertungsergebnisse.
        model_type: Typ des Modells (z.B. with_extra_conditions).
        detailed_results_path: Pfad zur CSV-Datei mit den detaillierten Metrikergebnissen.
    """

    analysis_dir = os.path.join(output_dir, model_type, 'condition_analysis')
    os.makedirs(analysis_dir, exist_ok=True)

    condition_prediction_path = os.path.join(analysis_dir, 'condition_prediction_mapping.csv')
    high_loss_samples_path = os.path.join(analysis_dir, 'high_loss_samples.csv')
    correlation_heatmap_path = os.path.join(analysis_dir, 'correlation_heatmap.png')
    condition_loss_distribution_path = os.path.join(analysis_dir, 'condition_loss_distribution.png')
    hypotheses_path = os.path.join(analysis_dir, 'hypotheses.txt')

    # Laden der Testdaten und Metrikergebnisse
    test_csv = pd.read_csv(os.path.join(data_dir, "test.csv"))
    metrics_df = pd.read_csv(detailed_results_path)

    merged_df = pd.merge(test_csv, metrics_df, on="sample_id", how="inner")

    # Konvertieren von 'db' in numerische Werte, falls vorhanden
    if 'db' in merged_df.columns:
      try:
          merged_df['db'] = merged_df['db'].apply(lambda x: eval(x)['lwd500'] if isinstance(x, str) and 'lwd500' in x else np.nan)
      except:
          merged_df['db'] = np.nan
          
    # Laden der Trainingsdaten, falls vorhanden
    train_csv_path = os.path.join(os.path.dirname(data_dir), "train", "train.csv")
    if os.path.exists(train_csv_path):
        train_df = pd.read_csv(train_csv_path)
        print(f"Trainingsdaten geladen von: {train_csv_path}")

        if 'db' in train_df.columns:
            try:
                train_df['db'] = train_df['db'].apply(lambda x: eval(x)['lwd500'] if isinstance(x, str) and 'lwd500' in x else np.nan)
            except:
                train_df['db'] = np.nan

            if 'osm' in train_df.columns:
                train_df['LoS_flag'] = train_df['osm'].str.contains('sight', case=False).astype(int)
            else:
                print("Die Spalte 'osm' ist in der train.csv nicht vorhanden.")
        else:
            print("Die Spalte 'db' ist in der train.csv nicht vorhanden.")
    else:
        train_df = None
        print("Keine train.csv Datei gefunden. Überspringe Analysen mit Trainingsdaten.")

    # Erstellen der condition_prediction_mapping.csv (erweitert um predicted_db_values)
    condition_prediction_mapping = []
    for index, row in tqdm(merged_df.iterrows(), total=len(merged_df), desc="Creating condition-prediction mapping"):
        pred_path = os.path.join(pred_dir, f"y_{index}.png")
        
        # Pixelweise dB-Werte aus PNG-Bild laden
        if os.path.exists(pred_path):
            try:
                img = Image.open(pred_path).convert("L")
                img_array = np.array(img)
                
                # Konvertiere Pixelwerte (0-255) in dB-Werte (0-100), Annahme: Schwarz(0) = 100dB, Weiß(255) = 0dB
                predicted_db_values = 100 - (img_array / 255.0) * 100
                
            except Exception as e:
              print(f"Fehler beim Laden oder Verarbeiten von Bild {pred_path}: {e}")
              predicted_db_values = np.full((256, 256), np.nan) # Erstelle ein 256x256 Array mit NaN Werten
        else:
            print(f"Bilddatei nicht gefunden: {pred_path}")
            predicted_db_values = np.full((256, 256), np.nan)

        condition_entry = {
            'sample_id': row.sample_id,
            'prediction_path': pred_path,
            'mae': row.MAE,
            'mape': row.MAPE,
            'LoS_MAE': row.LoS_MAE,
            'NLoS_MAE': row.NLoS_MAE,
            'LoS_wMAPE': row.LoS_wMAPE,
            'NLoS_wMAPE': row.NLoS_wMAPE,
            'predicted_db_values': predicted_db_values.tolist() # Speichere die dB-Werte als Liste
        }

        if 'temperature' in row:
            condition_entry['temperature'] = row.temperature
        if 'humidity' in row:
            condition_entry['humidity'] = row.humidity
        if 'db' in row:
            condition_entry['db'] = row.db
        if 'osm' in row:
            condition_entry['osm_path'] = row.osm

        condition_prediction_mapping.append(condition_entry)

    # Speichern der erweiterten condition_prediction_mapping.csv
    pd.DataFrame(condition_prediction_mapping).to_csv(condition_prediction_path, index=False)
    print(f"Condition-prediction mapping (mit predicted dB-Werten) saved to: {condition_prediction_path}")
    
    # Laden der condition_prediction_mapping.csv als DataFrame
    condition_prediction_df = pd.read_csv(condition_prediction_path)

    # Konvertiere die Spalte 'predicted_db_values' von String zurück zu NumPy Array
    condition_prediction_df['predicted_db_values'] = condition_prediction_df['predicted_db_values'].apply(lambda x: np.array(eval(x)))

    # Erstelle einen neuen DataFrame, der die relevanten Informationen enthält
    predictions_df = pd.DataFrame({
        'sample_id': condition_prediction_df['sample_id'],
        'LoS_wMAPE': condition_prediction_df['LoS_wMAPE'],
        'NLoS_wMAPE': condition_prediction_df['NLoS_wMAPE'],
        'predicted_db_values': condition_prediction_df['predicted_db_values']
    })
    
    # Funktion zum Visualisieren der Predictions mit true image und error map
    def visualize_predictions(df, title_prefix, output_dir, test_dir, merged_df):
      for index, row in df.iterrows():
          sample_id = row['sample_id']
          predicted_db_values = row['predicted_db_values']

          # Find corresponding row in merged_df using sample_id
          merged_row = merged_df[merged_df['sample_id'] == sample_id].iloc[0]

          # Load true soundmap using the path from merged_df
          true_soundmap_path = os.path.join(test_dir, merged_row['soundmap'].replace("./", ""))
          true_img = Image.open(true_soundmap_path).convert("L")
          true_img = true_img.resize((256, 256), Image.Resampling.NEAREST)
          true_soundmap = 100 - (np.array(true_img, dtype=np.float32)) / 255 * 100

          # Calculate absolute error
          abs_error = np.abs(predicted_db_values - true_soundmap)

          # Calculate wMAPE error per pixel
          # Vermeidung von Division durch Null
          
          # Erstelle eine Maske für Nullen in true_soundmap
          mask = true_soundmap != 0
          
          # Initialisiere wmape_error_map mit Nullen
          wmape_error_map = np.zeros_like(true_soundmap)
          
          # Berechne wmape_error_map nur für die Pixel, die in der Maske True sind
          wmape_error_map[mask] = np.abs((predicted_db_values[mask] - true_soundmap[mask]) / true_soundmap[mask])

          # Create figure with 4 subplots
          fig, axes = plt.subplots(1, 4, figsize=(24, 6))

          # Plot predicted soundmap
          im1 = axes[0].imshow(predicted_db_values, cmap='viridis', origin='lower', vmin=0, vmax=100)
          axes[0].set_title(f'{title_prefix} Prediction - Sample ID: {sample_id}')
          fig.colorbar(im1, ax=axes[0], label='dB')

          # Plot true soundmap
          im2 = axes[1].imshow(true_soundmap, cmap='viridis', origin='lower', vmin=0, vmax=100)
          axes[1].set_title(f'True Soundmap - Sample ID: {sample_id}')
          fig.colorbar(im2, ax=axes[1], label='dB')

          # Plot absolute error map
          im3 = axes[2].imshow(abs_error, cmap='RdYlGn_r', origin='lower', vmin=0)
          axes[2].set_title(f'Absolute Error - Sample ID: {sample_id}')
          fig.colorbar(im3, ax=axes[2], label='Absolute Error (dB)')

          # Plot wMAPE error map
          im4 = axes[3].imshow(wmape_error_map, cmap='RdYlGn_r', origin='lower', vmin=0)
          axes[3].set_title(f'wMAPE Error - Sample ID: {sample_id}')
          fig.colorbar(im4, ax=axes[3], label='wMAPE Error')

          plt.tight_layout()
          plt.savefig(os.path.join(output_dir, f'{title_prefix.lower().replace(" ", "_")}_prediction_analysis_{sample_id}.png'))
          plt.close(fig)

    # Sortiere den DataFrame nach LoS_wMAPE und NLoS_wMAPE, um die besten und schlechtesten Predictions zu finden
    best_los_predictions = predictions_df.sort_values(by='LoS_wMAPE', ascending=True).head(5)
    worst_los_predictions = predictions_df.sort_values(by='LoS_wMAPE', ascending=False).head(5)
    best_nlos_predictions = predictions_df.sort_values(by='NLoS_wMAPE', ascending=True).head(5)
    worst_nlos_predictions = predictions_df.sort_values(by='NLoS_wMAPE', ascending=False).head(5)

    # Visualisiere die besten und schlechtesten Predictions mit true image und error maps
    visualize_predictions(best_los_predictions, 'Best_LoS', analysis_dir, data_dir, merged_df)
    visualize_predictions(worst_los_predictions, 'Worst_LoS', analysis_dir, data_dir, merged_df)
    visualize_predictions(best_nlos_predictions, 'Best_NLoS', analysis_dir, data_dir, merged_df)
    visualize_predictions(worst_nlos_predictions, 'Worst_NLoS', analysis_dir, data_dir, merged_df)


    print(f"Beste und schlechteste Predictions visualisiert und gespeichert in: {analysis_dir}")

    # Identifizieren von Fällen mit hohen Verlustwerten (unverändert)
    high_loss_threshold_los = merged_df['LoS_wMAPE'].quantile(0.9)
    high_loss_threshold_nlos = merged_df['NLoS_wMAPE'].quantile(0.9)

    high_loss_samples = merged_df[
        (merged_df['LoS_wMAPE'] > high_loss_threshold_los) | (merged_df['NLoS_wMAPE'] > high_loss_threshold_nlos)
    ]
    high_loss_samples.to_csv(high_loss_samples_path, index=False)
    print(f"High loss samples saved to: {high_loss_samples_path}")

    # Korrelationsanalyse (unverändert)
    if 'temperature' in merged_df.columns and 'humidity' in merged_df.columns and 'db' in merged_df.columns:
        correlation_matrix = merged_df[['temperature', 'humidity', 'db', 'LoS_wMAPE', 'NLoS_wMAPE']].corr()
        plt.figure(figsize=(8, 6))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
        plt.title('Correlation Heatmap')
        plt.tight_layout()
        plt.savefig(correlation_heatmap_path)
        plt.close()
        print(f"Correlation heatmap saved to: {correlation_heatmap_path}")

    # Visualisierung der Verteilung der Verluste in Abhängigkeit von den Bedingungen (Scatterplots - unverändert)
    if 'temperature' in merged_df.columns:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=merged_df, x='temperature', y='LoS_wMAPE', label='LoS_wMAPE')
        sns.scatterplot(data=merged_df, x='temperature', y='NLoS_wMAPE', label='NLoS_wMAPE')
        plt.title('Loss Distribution vs. Temperature')
        plt.xlabel('Temperature')
        plt.ylabel('Loss (wMAPE)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(analysis_dir, 'temperature_loss_distribution.png'))
        plt.close()

    if 'humidity' in merged_df.columns:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=merged_df, x='humidity', y='LoS_wMAPE', label='LoS_wMAPE')
        sns.scatterplot(data=merged_df, x='humidity', y='NLoS_wMAPE', label='NLoS_wMAPE')
        plt.title('Loss Distribution vs. Humidity')
        plt.xlabel('Humidity')
        plt.ylabel('Loss (wMAPE)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(analysis_dir, 'humidity_loss_distribution.png'))
        plt.close()

    if 'db' in merged_df.columns:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=merged_df, x='db', y='LoS_wMAPE', label='LoS_wMAPE')
        sns.scatterplot(data=merged_df, x='db', y='NLoS_wMAPE', label='NLoS_wMAPE')
        plt.title('Loss Distribution vs. dB')
        plt.xlabel('dB')
        plt.ylabel('Loss (wMAPE)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(analysis_dir, 'db_loss_distribution.png'))
        plt.close()

    print(f"Condition-loss distribution plots saved to: {analysis_dir}")

    # Heatmaps für LoS_wMAPE und NLoS_wMAPE in Abhängigkeit von den Conditions
    conditions = ['temperature', 'humidity', 'db']
    for condition in conditions:
      if condition in merged_df.columns:
          # Diskretisierung der Conditions für die Heatmap
          merged_df[f'{condition}_binned'] = pd.cut(merged_df[condition], bins=10)

          # Erstellen der Heatmap für LoS_wMAPE
          heatmap_data_los = merged_df.pivot_table(index=f'{condition}_binned', values='LoS_wMAPE', aggfunc='mean', observed=False)
          plt.figure(figsize=(10, 6))
          sns.heatmap(heatmap_data_los, annot=True, cmap='coolwarm', fmt=".2f")
          plt.title(f'LoS_wMAPE Distribution vs. {condition} (Binned)')
          plt.ylabel(condition)
          plt.xlabel('LoS_wMAPE')
          plt.tight_layout()
          plt.savefig(os.path.join(analysis_dir, f'heatmap_LoS_wMAPE_{condition}.png'))
          plt.close()

          # Erstellen der Heatmap für NLoS_wMAPE
          heatmap_data_nlos = merged_df.pivot_table(index=f'{condition}_binned', values='NLoS_wMAPE', aggfunc='mean', observed=False)
          plt.figure(figsize=(10, 6))
          sns.heatmap(heatmap_data_nlos, annot=True, cmap='coolwarm', fmt=".2f")
          plt.title(f'NLoS_wMAPE Distribution vs. {condition} (Binned)')
          plt.ylabel(condition)
          plt.xlabel('NLoS_wMAPE')
          plt.tight_layout()
          plt.savefig(os.path.join(analysis_dir, f'heatmap_NLoS_wMAPE_{condition}.png'))
          plt.close()

    print(f"Heatmaps saved to: {analysis_dir}")

    #  Weiterführende Analyse 1: Scatterplots von vorhergesagten vs. tatsächlichen dB-Werten mit Fehlerkodierung
    if 'db' in merged_df.columns:
        # LoS-Fälle
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=merged_df[merged_df['LoS_wMAPE'].notna()], x='db', y='db', hue='LoS_wMAPE', palette='viridis', size='LoS_wMAPE')
        plt.plot([min(merged_df['db']), max(merged_df['db'])], [min(merged_df['db']), max(merged_df['db'])], linestyle='--', color='red')  # Diagonale
        plt.title('Vorhergesagte vs. Tatsächliche dB-Werte (LoS)')
        plt.xlabel('Tatsächliche dB')
        plt.ylabel('Vorhergesagte dB')
        plt.legend(title='LoS_wMAPE')
        plt.tight_layout()
        plt.savefig(os.path.join(analysis_dir, 'scatter_predicted_vs_actual_db_LoS.png'))
        plt.close()

        # NLoS-Fälle
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=merged_df[merged_df['NLoS_wMAPE'].notna()], x='db', y='db', hue='NLoS_wMAPE', palette='viridis', size='NLoS_wMAPE')
        plt.plot([min(merged_df['db']), max(merged_df['db'])], [min(merged_df['db']), max(merged_df['db'])], linestyle='--', color='red')  # Diagonale
        plt.title('Vorhergesagte vs. Tatsächliche dB-Werte (NLoS)')
        plt.xlabel('Tatsächliche dB')
        plt.ylabel('Vorhergesagte dB')
        plt.legend(title='NLoS_wMAPE')
        plt.tight_layout()
        plt.savefig(os.path.join(analysis_dir, 'scatter_predicted_vs_actual_db_NLoS.png'))
        plt.close()

        print(f"Scatterplots (vorhergesagte vs. tatsächliche dB) saved to: {analysis_dir}")

    # Weiterführende Analyse 2: Durchschnittliche Fehler in dB-Bereichen
    if 'db' in merged_df.columns:
        db_bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        merged_df['db_binned'] = pd.cut(merged_df['db'], bins=db_bins)

        error_metrics = ['MAE', 'MAPE', 'LoS_MAE', 'NLoS_MAE', 'LoS_wMAPE', 'NLoS_wMAPE']
        avg_errors = merged_df.groupby('db_binned', observed=False)[error_metrics].mean()

        print("\nDurchschnittliche Fehler in dB-Bereichen:")
        print(avg_errors)

        avg_errors.to_csv(os.path.join(analysis_dir, 'average_errors_db_ranges.csv'))
        print(f"Durchschnittliche Fehler in dB-Bereichen saved to: {os.path.join(analysis_dir, 'average_errors_db_ranges.csv')}")

    # Weiterführende Analyse 3: Verteilung der LoS/NLoS-Fälle in den Trainingsdaten
    if train_df is not None and 'db' in train_df.columns and 'LoS_flag' in train_df.columns:
        db_bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        plt.figure(figsize=(10, 6))
        sns.histplot(train_df, x='db', hue='LoS_flag', bins=db_bins, element='step', fill=False)
        plt.title('Verteilung der LoS/NLoS-Fälle in den Trainingsdaten nach dB-Wert')
        plt.xlabel('dB')
        plt.ylabel('Anzahl')
        plt.legend(title='LoS/NLoS', labels=['NLoS', 'LoS'])
        plt.tight_layout()
        plt.savefig(os.path.join(analysis_dir, 'distribution_LoS_NLoS_train_data.png'))
        plt.close()

        print(f"Verteilung der LoS/NLoS-Fälle in Trainingsdaten (Plot) saved to: {analysis_dir}")

        # Weiterführende Analyse 4: Vergleich der dB-Wert-Verteilungen zwischen Trainings- und Testdaten
        plt.figure(figsize=(10, 6))
        sns.histplot(train_df, x='db', label='Trainingsdaten', bins=db_bins, element='step', fill=False)
        sns.histplot(merged_df, x='db', label='Testdaten', bins=db_bins, element='step', fill=False, color='red')
        plt.title('Vergleich der dB-Wert-Verteilungen zwischen Trainings- und Testdaten')
        plt.xlabel('dB')
        plt.ylabel('Anzahl')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(analysis_dir, 'comparison_db_distribution_train_test.png'))
        plt.close()

        print(f"Vergleich der dB-Verteilungen (Plot) saved to: {analysis_dir}")

    # Weiterführende Analyse 5: Korrelation zwischen Trainings- und Testdaten für numerische 'conditions'
    if train_df is not None:
      numerical_conditions = ['temperature', 'humidity', 'db']
      for condition in numerical_conditions:
        if condition in merged_df.columns and condition in train_df.columns:
            correlation = train_df[condition].corr(merged_df[condition])
            print(f"\nKorrelation zwischen Trainings- und Testdaten für '{condition}': {correlation:.4f}")

            # Scatterplot Trainings- vs. Testdaten für die Condition
            plt.figure(figsize=(8, 6))
            sns.scatterplot(x=train_df[condition], y=merged_df[condition])
            plt.xlabel(f"Trainingsdaten: {condition}")
            plt.ylabel(f"Testdaten: {condition}")
            plt.title(f"Korrelation: {condition} (Trainings- vs. Testdaten)")

            # Find min and max across both datasets
            min_val = min(train_df[condition].min(), merged_df[condition].min())
            max_val = max(train_df[condition].max(), merged_df[condition].max())

            plt.plot([min_val, max_val], [min_val, max_val], linestyle='--', color='red')  # Diagonale
            plt.tight_layout()
            plt.savefig(os.path.join(analysis_dir, f'correlation_train_test_{condition}.png'))
            plt.close()

            print(f"Scatterplot (Trainings- vs. Testdaten) für '{condition}' saved to: {analysis_dir}")

    # Hypothesenfindung (angepasst, um leere Hypothesen zu vermeiden)
    hypotheses = []
    if 'temperature' in merged_df.columns:
        temp_corr_los = merged_df['temperature'].corr(merged_df['LoS_wMAPE'])
        temp_corr_nlos = merged_df['temperature'].corr(merged_df['NLoS_wMAPE'])
        if not np.isnan(temp_corr_los) and not np.isnan(temp_corr_nlos):
            hypotheses.append(f"Temperature correlation with LoS_wMAPE: {temp_corr_los:.4f}, with NLoS_wMAPE: {temp_corr_nlos:.4f}")

    if 'humidity' in merged_df.columns:
        humidity_corr_los = merged_df['humidity'].corr(merged_df['LoS_wMAPE'])
        humidity_corr_nlos = merged_df['humidity'].corr(merged_df['NLoS_wMAPE'])
        if not np.isnan(humidity_corr_los) and not np.isnan(humidity_corr_nlos):
            hypotheses.append(f"Humidity correlation with LoS_wMAPE: {humidity_corr_los:.4f}, with NLoS_wMAPE: {humidity_corr_nlos:.4f}")

    if 'db' in merged_df.columns:
        db_corr_los = merged_df['db'].corr(merged_df['LoS_wMAPE'])
        db_corr_nlos = merged_df['db'].corr(merged_df['NLoS_wMAPE'])
        if not np.isnan(db_corr_los) and not np.isnan(db_corr_nlos):
            hypotheses.append(f"dB correlation with LoS_wMAPE: {db_corr_los:.4f}, with NLoS_wMAPE: {db_corr_nlos:.4f}")

    if hypotheses:
        with open(hypotheses_path, 'w') as f:
            f.write('\n'.join(hypotheses))
        print(f"Hypotheses saved to: {hypotheses_path}")
    else:
        print("No hypotheses generated due to missing or invalid correlation data.")

    print(f"Condition analysis complete. Results saved to: {analysis_dir}")

def parse_args():
    parser = argparse.ArgumentParser(description="Analyze conditions impact on model predictions.")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing test dataset")
    parser.add_argument("--pred_dir", type=str, required=True,
                        help="Directory containing predictions to evaluate")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory for evaluation outputs")
    parser.add_argument("--model_type", type=str, required=True,
                        help="Type of model being evaluated (e.g., with_extra_conditions)")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    detailed_results_path = os.path.join(args.output_dir, args.model_type, 'metrics', 'detailed_results.csv')
    analyze_conditions(args.data_dir, args.pred_dir, args.output_dir, args.model_type, detailed_results_path)