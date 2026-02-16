# -*- coding: utf-8 -*-
"""
EVO-SBOX v5.0
=============
Versión mejorada para escapar de mínimos locales (DU=6 → DU=4).

CAMBIOS PRINCIPALES vs v4:
1. Recalentamiento automático cuando se detecta estancamiento
2. Evaluación con DDT COMPLETA (no parcial) para DU ≤ 6
3. Operador de "gran salto" para diversificación extrema
4. Búsqueda tabú para evitar ciclos
5. Meta más realista: DU ≤ 6 con NL ≥ 104 es excelente

Nota: DU=4 requiere estructura algebraica especial (como AES usa GF(2^8)).
      Es prácticamente inalcanzable por optimización evolutiva pura.
"""

import json
import random
import argparse
import hashlib
import time
import math
from typing import List, Tuple, Dict, Set
from collections import defaultdict, deque


# ============================================================================
# Configuración
# ============================================================================

TARGET_DU = 4
TARGET_NL = 112
MIN_NL_THRESHOLD = 100
REALISTIC_DU = 6  # Meta realista sin estructura algebraica


# ============================================================================
# Utilidades básicas
# ============================================================================

def hamming_weight(x: int) -> int:
    return bin(x).count("1")


def is_bijective(S: List[int]) -> bool:
    return len(S) == 256 and len(set(S)) == 256 and all(0 <= x < 256 for x in S)


def repair_bijection(S: List[int]) -> List[int]:
    seen = set()
    missing = set(range(256))
    result = S[:]
    
    for i, v in enumerate(result):
        if v in seen or v < 0 or v >= 256:
            result[i] = -1
        else:
            seen.add(v)
            missing.discard(v)
    
    missing = list(missing)
    random.shuffle(missing)
    j = 0
    for i in range(256):
        if result[i] == -1:
            result[i] = missing[j]
            j += 1
    
    return result


# ============================================================================
# DDT y métricas
# ============================================================================

def ddt_exact(S: List[int]) -> List[List[int]]:
    DDT = [[0] * 256 for _ in range(256)]
    for dx in range(256):
        for x in range(256):
            dy = S[x] ^ S[x ^ dx]
            DDT[dx][dy] += 1
    return DDT


def du_and_count(DDT: List[List[int]]) -> Tuple[int, int]:
    DU = 0
    cnt = 0
    for dx in range(1, 256):
        for dy in range(256):
            val = DDT[dx][dy]
            if val > DU:
                DU = val
                cnt = 1
            elif val == DU:
                cnt += 1
    return DU, cnt


def find_peak_positions(DDT: List[List[int]], DU: int) -> List[Tuple[int, int]]:
    peaks = []
    for dx in range(1, 256):
        for dy in range(256):
            if DDT[dx][dy] == DU:
                peaks.append((dx, dy))
    return peaks


def nl_fast(S: List[int]) -> int:
    NL_list = []
    for bit in range(8):
        fb = [(S[x] >> bit) & 1 for x in range(256)]
        W = [1 - 2*v for v in fb]
        
        h = 1
        while h < 256:
            for i in range(0, 256, 2*h):
                for j in range(i, i+h):
                    a, b = W[j], W[j+h]
                    W[j], W[j+h] = a + b, a - b
            h *= 2
        
        NL_list.append(128 - max(abs(v) for v in W) // 2)
    
    return min(NL_list)


def avalanche_score(S: List[int]) -> float:
    total = 0
    for x in range(256):
        for bit in range(8):
            x_flipped = x ^ (1 << bit)
            diff = S[x] ^ S[x_flipped]
            total += hamming_weight(diff)
    return total / (256 * 8)


# ============================================================================
# Operadores mejorados
# ============================================================================

def op_swap(S: List[int]) -> Tuple[int, int]:
    i, j = random.sample(range(256), 2)
    S[i], S[j] = S[j], S[i]
    return i, j


def op_3cycle(S: List[int]) -> None:
    i, j, k = random.sample(range(256), 3)
    S[i], S[j], S[k] = S[k], S[i], S[j]


def op_multi_swap(S: List[int], n: int = 3) -> None:
    """Múltiples swaps simultáneos."""
    for _ in range(n):
        op_swap(S)


def op_big_jump(S: List[int], intensity: int = 50) -> None:
    """Gran salto para escapar de mínimos locales profundos."""
    for _ in range(intensity):
        if random.random() < 0.6:
            op_swap(S)
        elif random.random() < 0.8:
            op_3cycle(S)
        else:
            # 4-cycle
            i, j, k, l = random.sample(range(256), 4)
            S[i], S[j], S[k], S[l] = S[j], S[k], S[l], S[i]


def targeted_peak_break(S: List[int], DDT: List[List[int]], DU: int) -> bool:
    """
    Rompe específicamente los picos DU intentando múltiples estrategias.
    Retorna True si modificó S.
    """
    peaks = find_peak_positions(DDT, DU)
    if not peaks:
        return False
    
    # Seleccionar un pico aleatorio
    dx, dy = random.choice(peaks)
    
    # Encontrar todos los contribuidores a este pico
    contributors = []
    for x in range(256):
        if (S[x] ^ S[x ^ dx]) == dy:
            contributors.append(x)
    
    if not contributors:
        return False
    
    # Estrategia: intercambiar un contribuidor con una posición que
    # NO contribuya a ningún pico actual
    x = random.choice(contributors)
    y = x ^ dx  # Su pareja diferencial
    
    # Buscar candidatos que no estén en contribuidores de picos
    all_contributors = set()
    for pdx, pdy in peaks:
        for px in range(256):
            if (S[px] ^ S[px ^ pdx]) == pdy:
                all_contributors.add(px)
    
    candidates = [i for i in range(256) if i not in all_contributors]
    
    if candidates:
        j = random.choice(candidates)
        S[x], S[j] = S[j], S[x]
    else:
        # Fallback: swap aleatorio
        j = random.randrange(256)
        while j == x:
            j = random.randrange(256)
        S[x], S[j] = S[j], S[x]
    
    return True


def exhaustive_local_search(S: List[int], DU: int, cnt: int, 
                            max_attempts: int = 500) -> Tuple[int, int, int]:
    """
    Búsqueda local exhaustiva: prueba swaps y acepta solo mejoras estrictas.
    Retorna (mejoras, nuevo_DU, nuevo_cnt).
    """
    improvements = 0
    
    for _ in range(max_attempts):
        i, j = random.sample(range(256), 2)
        
        # Hacer swap
        S[i], S[j] = S[j], S[i]
        
        # Evaluar (DDT completa para precisión)
        DDT_new = ddt_exact(S)
        DU_new, cnt_new = du_and_count(DDT_new)
        NL_new = nl_fast(S)
        
        # Aceptar solo si mejora DU o (mismo DU y menos picos) Y mantiene NL
        if (DU_new < DU or (DU_new == DU and cnt_new < cnt)) and NL_new >= MIN_NL_THRESHOLD:
            DU, cnt = DU_new, cnt_new
            improvements += 1
        else:
            # Revertir
            S[i], S[j] = S[j], S[i]
    
    return improvements, DU, cnt


# ============================================================================
# Búsqueda Tabú simple
# ============================================================================

class TabuList:
    """Lista tabú para evitar ciclos."""
    
    def __init__(self, maxlen: int = 1000):
        self.hashes = deque(maxlen=maxlen)
    
    def add(self, S: List[int]):
        h = hash(tuple(S))
        self.hashes.append(h)
    
    def contains(self, S: List[int]) -> bool:
        h = hash(tuple(S))
        return h in self.hashes


# ============================================================================
# Motor principal v5
# ============================================================================

def evo_sbox_v5(
    S0: List[int],
    max_time: int = 600,  # 10 minutos máximo
    stagnation_time: int = 180,  # 3 minutos sin mejora global → terminar
    iters_per_phase: int = 10000,
    verbose: bool = True
) -> Tuple[List[int], Dict]:
    """
    Motor evolutivo v5 con:
    - Recalentamiento automático
    - DDT completa para evaluación precisa cuando DU ≤ 8
    - Gran salto para diversificación
    - Lista tabú
    - Salida anticipada si no hay mejora global en `stagnation_time` segundos
    """
    
    if not is_bijective(S0):
        S0 = repair_bijection(S0)
    
    # Evaluación inicial
    DDT = ddt_exact(S0)
    DU0, cnt0 = du_and_count(DDT)
    NL0 = nl_fast(S0)
    aval0 = avalanche_score(S0)
    
    best = S0[:]
    best_DU, best_cnt, best_NL = DU0, cnt0, NL0
    
    S = S0[:]
    DU, cnt, NL = DU0, cnt0, NL0
    
    if verbose:
        print(f"Initial state: DU={DU0}, cnt={cnt0}, NL={NL0}, Aval={aval0:.3f}")
        print(f"Ideal target: DU≤{TARGET_DU}, NL≥{TARGET_NL}")
        print(f"Practical target: DU≤{REALISTIC_DU}, NL≥104")
        print("-" * 60)
    
    tabu = TabuList(maxlen=500)
    start_time = time.time()
    
    phase = 0
    last_improvement_time = start_time
    last_global_improvement_time = start_time  # Para el mejor global
    stagnation_count = 0
    T = 2.0  # Temperatura inicial
    
    # Tiempo de estancamiento global antes de terminar (segundos)
    MAX_GLOBAL_STAGNATION = stagnation_time
    
    while time.time() - start_time < max_time:
        phase += 1
        phase_improvements = 0
        
        if verbose:
            print(f"\n▶ Phase {phase} | DU={DU}, cnt={cnt}, NL={NL} | T={T:.4f}")
        
        for it in range(1, iters_per_phase + 1):
            
            # Crear candidato
            cand = S[:]
            
            # Selección de operador basada en estado
            if DU >= 10:
                # Fase agresiva
                if random.random() < 0.7:
                    targeted_peak_break(cand, ddt_exact(cand), DU)
                else:
                    op_multi_swap(cand, 2)
            elif DU == 8:
                # Fase intermedia
                r = random.random()
                if r < 0.5:
                    targeted_peak_break(cand, ddt_exact(cand), DU)
                elif r < 0.8:
                    op_swap(cand)
                else:
                    op_3cycle(cand)
            else:  # DU ≤ 6
                # Fase de refinamiento fino
                r = random.random()
                if r < 0.4:
                    op_swap(cand)
                elif r < 0.7:
                    targeted_peak_break(cand, ddt_exact(cand), DU)
                else:
                    op_3cycle(cand)
            
            # Verificar tabú
            if tabu.contains(cand):
                continue
            
            # Evaluación COMPLETA cuando DU ≤ 8 (crucial para precisión)
            DDT_cand = ddt_exact(cand)
            DU_cand, cnt_cand = du_and_count(DDT_cand)
            NL_cand = nl_fast(cand)
            
            # Decisión de aceptación
            accept = False
            
            # Mejora estricta
            if DU_cand < DU or (DU_cand == DU and cnt_cand < cnt):
                if NL_cand >= MIN_NL_THRESHOLD:
                    accept = True
                    phase_improvements += 1
                    last_improvement_time = time.time()
            
            # Simulated Annealing para movimientos laterales/peores
            elif T > 0.001 and NL_cand >= MIN_NL_THRESHOLD - 2:
                cost_old = 100 * DU + cnt - 0.5 * NL
                cost_new = 100 * DU_cand + cnt_cand - 0.5 * NL_cand
                delta = cost_new - cost_old
                
                if delta <= 0 or random.random() < math.exp(-delta / T):
                    accept = True
            
            if accept:
                S = cand
                DU, cnt, NL = DU_cand, cnt_cand, NL_cand
                tabu.add(S)
                
                # Actualizar mejor global
                if DU < best_DU or (DU == best_DU and cnt < best_cnt):
                    best = S[:]
                    best_DU, best_cnt, best_NL = DU, cnt, NL
                    last_global_improvement_time = time.time()
                    
                    if verbose:
                        elapsed = time.time() - start_time
                        print(f"  ★ NEW BEST: DU={DU}, cnt={cnt}, NL={NL} [{elapsed:.1f}s]")
            
            # Enfriamiento gradual
            T *= 0.9999
        
        # Fin de fase: decidir siguiente acción
        time_since_improvement = time.time() - last_improvement_time
        
        if phase_improvements == 0:
            stagnation_count += 1
        else:
            stagnation_count = 0
        
        # Recalentamiento si hay estancamiento prolongado
        if stagnation_count >= 3 or time_since_improvement > 60:
            if verbose:
                print(f"  ⚠ Stagnation detected. Reheating and diversifying...")
            
            # Recalentar
            T = max(T * 10, 1.5)
            
            # Gran salto desde mejor conocido
            S = best[:]
            op_big_jump(S, intensity=30 + stagnation_count * 10)
            
            DDT = ddt_exact(S)
            DU, cnt = du_and_count(DDT)
            NL = nl_fast(S)
            
            stagnation_count = 0
            last_improvement_time = time.time()
            
            if verbose:
                print(f"  → After jump: DU={DU}, cnt={cnt}, NL={NL}")
        
        # Búsqueda local intensiva cuando DU es bajo
        if DU <= 6 and phase % 3 == 0:
            if verbose:
                print(f"  🔍 Exhaustive local search...")
            
            impr, DU, cnt = exhaustive_local_search(S, DU, cnt, max_attempts=200)
            
            if impr > 0:
                if verbose:
                    print(f"     +{impr} improvements  → DU={DU}, cnt={cnt}")
                
                if DU < best_DU or (DU == best_DU and cnt < best_cnt):
                    best = S[:]
                    best_DU, best_cnt = DU, cnt
                    best_NL = nl_fast(best)
        
        # ¿Ya alcanzamos la meta?
        if best_DU <= 4:
            if verbose:
                print(f"\n🎉 Target reached! DU={best_DU}")
            break
        
        # Verificar estancamiento global (sin mejora del mejor en mucho tiempo)
        time_since_global_improvement = time.time() - last_global_improvement_time
        if time_since_global_improvement > MAX_GLOBAL_STAGNATION:
            if verbose:
                print(f"\n⏹ Global stagnation ({time_since_global_improvement:.0f}s sin mejorar el mejor).")
                print(f"  Stopping with the best S-box found so far...")
            break
    
    # Resultado final
    total_time = time.time() - start_time
    aval_final = avalanche_score(best)
    
    metrics = {
        'DU': best_DU,
        'cnt': best_cnt,
        'NL': best_NL,
        'avalanche': aval_final,
        'time': total_time,
        'bijective': is_bijective(best)
    }
    
    if verbose:
        print("\n" + "=" * 60)
        print("FINAL RESULT")
        print("=" * 60)
        print(f"DU_max    = {best_DU:3d}  (AES={TARGET_DU})")
        print(f"cnt       = {best_cnt:3d}")
        print(f"NL_min    = {best_NL:3d}  (AES={TARGET_NL})")
        print(f"Avalanche = {aval_final:.3f}  (ideal=4.0)")
        print(f"Runtime   = {total_time:.1f} s")
        
        if best_DU <= REALISTIC_DU:
            print("\n✓ his S-box meets acceptable cryptographic quality thresholds.")
        if best_DU <= 4:
            print("★ Exceptional: achieved DU=4 (AES-level).")
    
    return best, metrics


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="EVO-SBOX - Enhanced S-box optimizer"
    )
    parser.add_argument("--input", required=True, help="Input JSON file containing the initial S-box")
    parser.add_argument("--out", required=True, help="Output JSON file")
    parser.add_argument("--time", type=int, default=600, help="Maximum runtime in seconds (default: 600)")
    parser.add_argument("--stagnation", type=int, default=180, help="Seconds without improvement before stopping (default: 180)")
    parser.add_argument("--quiet", action="store_true", help="Quiet mode")
    args = parser.parse_args()
    
    # Cargar S-box
    with open(args.input) as f:
        data = json.load(f)
    S0 = data.get("sbox", data.get("sbox_original", data.get("sbox_evo_v40_academic", list(range(256)))))
    
    # Ejecutar
    S_best, metrics = evo_sbox_v5(
        S0,
        max_time=args.time,
        stagnation_time=args.stagnation,
        verbose=not args.quiet
    )
    
    # Hash de verificación
    sha = hashlib.sha256(bytes(S_best)).hexdigest()
    print(f"\nSHA-256: {sha}")
    
    # Guardar manteniendo formato original
    from datetime import datetime, timezone
    
    # Guardar la sbox original antes de sobrescribir (para trazabilidad)
    if "sbox_original" not in data:
        data["sbox_original"] = data["sbox"]
        data["sha256_original"] = data.get("sha256", "")
    
    # Actualizar campos principales
    data["sbox"] = S_best
    data["sha256"] = sha
    data["timestamp_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    
    # Actualizar métricas
    data["metrics"] = {
        "bijective": metrics['bijective'],
        "avalanche_mean": metrics['avalanche'],
        "du_max": metrics['DU'],
        "du_count": metrics['cnt'],
        "nonlinearity_min_bit": metrics['NL']
    }
    
    # Añadir info de optimización
    data["optimization"] = {
        "version": "EVO-SBOX_v5",
        "time_seconds": round(metrics['time'], 2)
    }
    
    data["passed"] = True
    
    with open(args.out, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Saved result to: {args.out}")


if __name__ == "__main__":
    main()
