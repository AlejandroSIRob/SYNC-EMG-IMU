"""
FUSION MULTIMODAL CIENTÍFICA (V - Integrated Library + IMU)
-------------------------------------------------------------
Autor: Alejandro Solar Iglesias
Mejora Final:
  - Mantiene la lógica V39 (EMG perfecto con librería).
  - Añade reporte gráfico para IMUs (Aceleración).
  - Todo sincronizado al 3º impacto.
Para ejecuatar: 
  - python Sync.py --ruta_toma "ruta/a/la/toma" (o con --emg y --imu para archivos específicos)
  - python .\Sync.py --emg .\Data\EMG\2026-02-04-19-37_v1.csv --imu .\Data\IMU\cinematica_v1.sto      

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
import os
import glob
import sys
import re
import argparse

# =========================================================
# 0. AÑADIR LA CARPETA ANTERIOR AL PATH DE PYTHON
# =========================================================
# Calculamos la ruta absoluta de la carpeta "padre" (un nivel arriba)
ruta_padre = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Si la ruta no está en las rutas del sistema, la añadimos
if ruta_padre not in sys.path:
    sys.path.append(ruta_padre)

# =========================================================
# 1. IMPORTACIÓN DE TU LIBRERÍA
# =========================================================
USE_LIBRARY = False
try:
    import noraxon_analytics as na
    print("[INFO] Librería 'noraxon_analytics' cargada correctamente.")
    FS_EMG = na.DEFAULT_FS
    USE_LIBRARY = True
except ImportError:
    print("[WARN] No se encontró 'noraxon_analytics.py'. Usando modo manual.")
    from scipy.signal import butter, filtfilt # Solo se importan si no hay librería
    FS_EMG = 2000.0 
    USE_LIBRARY = False

# --- CONFIGURACIÓN ---
FS_MASTER = 100.0           
WINDOW_SEARCH = [2.0, 60.0] 
CLUSTER_MAX_DURATION = 2.5  
MIN_PEAK_DIST_S = 0.15      

# =========================================================
# 2. PROCESAMIENTO EMG (CONECTADO A TU LIBRERÍA)
# =========================================================
def procesar_canal_emg(raw, fs):
        clean = na.remove_dc_offset(raw)
        filt = na.butter_bandpass_filter(clean, 20, 450, fs, order=4)
        env = na.compute_linear_envelope(filt, fs, cutoff=6, order=4)
        return env

def obtener_datos_emg(df):
    if df['time'].dtype == object:
        t = df['time'].str.replace(',', '.').astype(float).values
    else:
        t = df['time'].values
    cols = [c for c in df.columns if '(uV)' in c and 'Switch' not in c]
    senales = {}
    composite = np.zeros(len(t))
    for col in cols:
        raw = df[col].astype(float).values
        env = procesar_canal_emg(raw, FS_EMG)
        senales[col] = env
        if np.max(env) > 0: composite += (env / np.max(env))
    return senales, composite, t, cols

# =========================================================
# 3. PROCESAMIENTO IMU 
# =========================================================
def procesar_imu_acc(df, col_name):
    """Calcula la aceleración resultante (Jerk) del sensor IMU."""
    try:
        val = df[col_name].values
        if df[col_name].dtype == object:
            val = df[col_name].str.split(',', expand=True)[0].astype(float).values
        
        # Derivada segunda (Aceleración/Jerk aprox)
        vel = np.diff(val, prepend=val[0])
        acc = np.diff(vel, prepend=vel[0])
        return np.abs(acc)
    except:
        return np.zeros(len(df))

# =========================================================
# 4. UTILIDADES Y MOTORES
# =========================================================
def extraer_nombre_real(header_csv):
    limpio = header_csv.replace('(uV)', '').replace('[uV]', '').strip()
    if '.' in limpio:
        partes = limpio.split('.')
        candidatos = [p for p in partes if 'Ultium' not in p and 'EMG' not in p and len(p) > 1]
        if candidatos: return candidatos[0].strip()
    match = re.search(r'([A-Za-z\s]+)', limpio)
    if match and len(match.group(1)) > 3: return match.group(1).strip()
    return limpio[:30]

# =========================================================
# MOTOR DE DETECCIÓN (BASED ON FEATURE EXTRACTION)
# =========================================================
def detectar_tercer_impacto(senal, tiempo, nombre_sensor, fs, verbose=True):
    """
    Implementa un algoritmo de Sincronización Basada en Eventos (Event-Based Synchronization).
    
    METODOLOGÍA:
    1. Normalización Min-Max: Estandariza la amplitud para procesar Fuerza (N), EMG (uV) y Acc (m/s2) por igual.
    2. Detección de Picos (Local Maxima): Utiliza 'scipy.signal.find_peaks'.
    3. Periodo Refractario (150ms): Basado en la fisiología muscular para evitar dobles conteos.
    4. Umbral Adaptativo (Iterative Thresholding):
       - Inspirado en el método de Bonato et al. (1998) pero simplificado para picos discretos.
       - Realiza barridos de sensibilidad (40%, 25%, 15%, 5%) hasta encontrar el patrón.
    5. Validación de Clúster: Filtro de coherencia temporal (3 eventos en < 2.5s).
    
    RETORNO:
    - float: Tiempo exacto (s) del 3º evento (Final de la fase de sincronización).
    """
    # 1. Pre-procesado y Normalización
    senal = np.nan_to_num(senal)
    offset = np.mean(senal[:int(fs)]) # Tara inicial
    senal_clean = np.abs(senal - offset)
    mx = np.max(senal_clean)
    if mx == 0: return None
    senal_norm = senal_clean / mx

    # 2. Barrido de Umbral Adaptativo (Iterative Thresholding)
    # Buscamos el patrón reduciendo la exigencia progresivamente
    for sensibilidad in [0.4, 0.25, 0.1, 0.05]:
        # 3. Detección de Picos con Periodo Refractario (distance)
        peaks, _ = find_peaks(senal_norm, height=sensibilidad, distance=int(MIN_PEAK_DIST_S * fs))
        
        # 4. Validación de Clúster (Pattern Matching)
        if len(peaks) >= 3:
            for i in range(len(peaks) - 2):
                t1 = tiempo[peaks[i]]
                t3 = tiempo[peaks[i+2]]
                
                # Criterio de Coherencia Temporal
                if (t3 - t1) < CLUSTER_MAX_DURATION:
                    if verbose: print(f"   [{nombre_sensor}] Patrón detectado (Sensibilidad {sensibilidad}). T(3º) = {t3:.3f}s")
                    return t3

    # Fallback: Si no hay patrón claro, usar el Máximo Absoluto (Worst Case)
    idx_max = np.argmax(senal_norm)
    t_max = tiempo[idx_max]
    if verbose: print(f"   [{nombre_sensor}] WARNING: Patrón no convergente. Usando Máximo Absoluto: {t_max:.3f}s")
    return t_max

# Nota: Eliminado el procesamiento de fuerza — sincronización basada solo en EMG e IMU

def procesar_xsens_mano(df):
    """Solo para detectar el golpe (Hand Jerk)."""
    t = df['time'].values
    col_hand = next((c for c in df.columns if 'hand' in c.lower() or 'mano' in c.lower()), None)
    if not col_hand: return np.zeros(len(t)), t
    return procesar_imu_acc(df, col_hand), t

# =========================================================
# 5. GENERACIÓN DE REPORTES GRÁFICOS
# =========================================================
def generar_reporte_emg(df, emg_cols_originales, folder):
    num_musculos = len(emg_cols_originales)
    if num_musculos == 0:
        return
    fig, axes = plt.subplots(num_musculos, 1, figsize=(12, 3 + 2*num_musculos), sharex=True)
    if num_musculos == 1: axes = [axes]

    for i, col_orig in enumerate(emg_cols_originales):
        ax = axes[i]
        nombre_real = extraer_nombre_real(col_orig)
        col_match = None
        for c in df.columns:
            if 'EMG_' in c:
                nombre_col_df = c.replace('EMG_', '').replace('_', ' ')
                if nombre_real in nombre_col_df or nombre_col_df in nombre_real:
                    col_match = c; break
        if not col_match and i < len([c for c in df.columns if 'EMG_' in c]):
             posibles = [c for c in df.columns if 'EMG_' in c]
             col_match = posibles[i]

        if col_match:
            ax.plot(df['Time'], df[col_match], '#2ecc71', lw=1.2)
            ax.set_title(f"{nombre_real}", loc='left', fontweight='bold', fontsize=9, color='#145a32')
            ax.set_ylabel("uV", fontsize=8)
            ax.grid(True, alpha=0.3); ax.axvline(0, color='r', ls='--', alpha=0.6)
        else: ax.text(0.5, 0.5, "No encontrado", ha='center')

    axes[-1].set_xlabel("Tiempo (s)"); plt.tight_layout()
    plt.savefig(os.path.join(folder, "REPORTE_EMG.png"), dpi=100); plt.close()

def generar_reporte_imu(df, imu_cols, folder):
    """Genera gráfico de todas las IMUs."""
    n = len(imu_cols)
    if n == 0: return
    fig, axes = plt.subplots(n, 1, figsize=(12, 3 + 2*n), sharex=True)
    if n == 1: axes = [axes]

    # IMUs
    for i, col in enumerate(imu_cols):
        ax = axes[i]
        nombre = col.replace('_imu', '').replace('_', ' ').title()
        ax.plot(df['Time'], df[col], '#2980b9', lw=1.2)
        ax.set_title(f"IMU: {nombre}", loc='left', fontweight='bold', fontsize=9, color='#154360')
        ax.set_ylabel("Acc", fontsize=8)
        ax.grid(True, alpha=0.3); ax.axvline(0, color='r', ls='--', alpha=0.6)

    axes[-1].set_xlabel("Tiempo (s)"); plt.tight_layout()
    plt.savefig(os.path.join(folder, "REPORTE_IMU.png"), dpi=100); plt.close()

# =========================================================
# 6. MAIN
# =========================================================
def main():
    parser = argparse.ArgumentParser(description='Sync EMG and IMU (accepts folder or explicit files)')
    parser.add_argument('ruta_toma', nargs='?', default=None, help='Carpeta de la toma que contiene EMG/PROCESADO-Xsens')
    parser.add_argument('--emg', help='Ruta al fichero EMG .csv (opcional)')
    parser.add_argument('--imu', help='Ruta al fichero IMU .sto/.txt (opcional)')
    args = parser.parse_args()

    # Si se pasan rutas explícitas, úsalas
    if args.emg and args.imu:
        e_path = args.emg
        k_path = args.imu
        ruta_toma = os.path.dirname(e_path) or '.'
        print(f"\nANÁLISIS V44 (EMG+IMU) - archivos explícitos:\n EMG={e_path}\n IMU={k_path}\n")
    else:
        if not args.ruta_toma:
            parser.print_help()
            return
        ruta_toma = args.ruta_toma
        print(f"\nANÁLISIS V44 (EMG+IMU): {os.path.basename(ruta_toma)}\n")
        try:
            e_path = glob.glob(os.path.join(ruta_toma, "EMG", "*.csv"))[0]
            k_path = glob.glob(os.path.join(ruta_toma, "PROCESADO-Xsens", "*.sto"))[0]
        except IndexError:
            print("[ERROR] No se encontraron archivos EMG o XSens en la toma.")
            return

    # PROCESADO BASE
    df_e = pd.read_csv(e_path, sep=';', decimal=',', skiprows=3, engine='python')
    df_e.columns = [c.replace('"', '').strip() for c in df_e.columns]
    dict_emg, sig_e_comp, t_e, emg_cols_orig = obtener_datos_emg(df_e)
    df_k = pd.read_csv(k_path, sep='\t', skiprows=5)
    t_k = df_k['time'].values
    sig_k, _ = procesar_xsens_mano(df_k) # Solo para sync

    # SYNC (solo EMG y XSens)
    mask_e = (t_e > WINDOW_SEARCH[0]) & (t_e < WINDOW_SEARCH[1])
    mask_k = (t_k > WINDOW_SEARCH[0]) & (t_k < WINDOW_SEARCH[1])

    t_sync_e = detectar_tercer_impacto(sig_e_comp[mask_e], t_e[mask_e], "EMG_GLOBAL", FS_EMG)
    t_sync_k = detectar_tercer_impacto(sig_k[mask_k], t_k[mask_k], "XSENS", 60.0)

    if t_sync_e is None and t_sync_k is None:
        print("[WARN] No se detectó evento de sincronización en EMG ni XSens.")
        return

    # Referencia temporal: preferir EMG si está disponible, sino XSens
    t_sync_ref = t_sync_e if t_sync_e is not None else t_sync_k

    # DIAGNOSTICO EMG
    print(f"{'MÚSCULO REAL':<40} | {'EMD (ms)':<10} | {'ESTADO'}")
    print("-" * 80)
    for col in emg_cols_orig:
        nombre_real = extraer_nombre_real(col)
        if mask_e.any():
            t_pico = detectar_tercer_impacto(dict_emg[col][mask_e], t_e[mask_e], nombre_real, FS_EMG, verbose=False)
        else:
            t_pico = None
        if t_pico is not None and t_sync_e is not None:
            val = (t_pico - t_sync_e) * 1000.0
            print(f"{nombre_real:<40} | {val:+.1f} ms   | {'SYNC' if abs(val)<50 else 'DESFASE'}")
        else:
            print(f"{nombre_real:<40} | {'N/A':<10} | NO DATA/NO SYNC")
    # FUSIÓN: crear línea temporal maestra basada en la referencia detectada
    t_fin = max(t_e[-1], t_k[-1]) - t_sync_ref
    t_master = np.arange(-3.0, t_fin, 1/FS_MASTER)
    df_out = pd.DataFrame({'Time': t_master})
    remap = lambda t, y, t0: interp1d(t - t0, y, bounds_error=False, fill_value=0)(t_master)

    # 1. EMG
    for col in emg_cols_orig:
        nombre_real = extraer_nombre_real(col)
        safe_col = f'EMG_{nombre_real.replace(" ", "_")}'
        t0 = t_sync_e if t_sync_e is not None else t_sync_ref
        df_out[safe_col] = remap(t_e, dict_emg[col], t0)

    # 2. IMU (procesar todas las columnas IMU)
    imu_cols_orig = [c for c in df_k.columns if 'imu' in c]
    for col in imu_cols_orig:
        acc = procesar_imu_acc(df_k, col)
        t0k = t_sync_k if t_sync_k is not None else t_sync_ref
        df_out[col] = remap(t_k, acc, t0k)

    # GUARDAR
    out_folder = os.path.join(ruta_toma, "PROCESADO_COMPLETO")
    if not os.path.exists(out_folder): os.makedirs(out_folder)
    df_out.to_csv(os.path.join(out_folder, "DATASET_MAESTRO.csv"), index=False)

    # REPORTES
    print("-> Generando gráficas EMG...")
    generar_reporte_emg(df_out, emg_cols_orig, out_folder)
    print("-> Generando gráficas IMU...")
    generar_reporte_imu(df_out, imu_cols_orig, out_folder)
    
    print(f"[OK] {os.path.basename(ruta_toma)} finalizada.")

if __name__ == "__main__":
    main()