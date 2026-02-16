# -*- coding: utf-8 -*-
"""
sbox_randomorg_signed.py
------------------------
Genera una S-box 8x8 a partir de aleatoriedad física usando la **Signed API** de RANDOM.ORG
y **verifica criptográficamente** las firmas de los datos recibidos (vía método verifySignature).

Si alguna métrica cae fuera de los criterios definidos, la caja se descarta
y se vuelve a generar automáticamente hasta obtener una válida.

Uso:
    export RANDOM_ORG_API_KEY="..."
    python sbox_randomorg_signed.py --out sbox_artifact_signed.json
"""

import os, json, time, hashlib, argparse
from typing import List, Tuple, Dict, Any
import requests

# URL base de la API JSON-RPC de RANDOM.ORG
RNG_URL = "https://api.random.org/json-rpc/4/invoke"


# ==== Excepciones ====

class RandomOrgError(Exception):
    pass

class SignatureVerificationError(Exception):
    pass


# ==== Funciones de comunicación con RANDOM.ORG ====

def _rpc(method: str, params: Dict[str, Any], request_id: int = 1) -> Dict[str, Any]:
    payload = {"jsonrpc": "2.0", "method": method, "params": params, "id": request_id}
    resp = requests.post(RNG_URL, json=payload, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    if "error" in data:
        raise RandomOrgError(str(data["error"]))
    return data["result"]

def randomorg_generate_signed_integers(api_key: str, n: int, minv: int, maxv: int,
                                       replacement: bool, request_id: int) -> Tuple[list, dict]:
    res = _rpc("generateSignedIntegers", {
        "apiKey": api_key,
        "n": n,
        "min": minv,
        "max": maxv,
        "replacement": replacement
    }, request_id=request_id)

    rnd_obj = res.get("random")
    signature = res.get("signature")
    if rnd_obj is None or signature is None:
        raise RandomOrgError("La Signed API no devolvió 'random' o 'signature'.")

    data = rnd_obj.get("data")
    if data is None:
        raise RandomOrgError("Objeto 'random' sin campo 'data'.")

    bundle = {
        "random": rnd_obj,
        "signature": signature,
        "bitsLeft": res.get("bitsLeft"),
        "requestsLeft": res.get("requestsLeft"),
        "completionTime": rnd_obj.get("completionTime"),
        "requestId": request_id
    }
    return data, bundle

def randomorg_verify_signature(signed_random: dict, signature: str, request_id: int = 999) -> bool:
    res = _rpc("verifySignature", {
        "random": signed_random,
        "signature": signature
    }, request_id=request_id)

    if isinstance(res, bool):
        return bool(res)
    for key in ("authenticity", "valid", "verified"):
        if key in res:
            return bool(res[key])
    raise SignatureVerificationError(f"Respuesta inesperada de verifySignature: {res!r}")


# ==== Operaciones en GF(2) ====

def gf2_rank_and_inverse(mat: List[List[int]]):
    n=8; a=[row[:] for row in mat]; inv=[[1 if i==j else 0 for j in range(n)] for i in range(n)]
    rank=0; col=0; row=0
    while row<n and col<n:
        pivot=-1
        for r in range(row,n):
            if a[r][col]==1: pivot=r; break
        if pivot==-1: col+=1; continue
        a[row],a[pivot]=a[pivot],a[row]; inv[row],inv[pivot]=inv[pivot],inv[row]
        for r in range(n):
            if r!=row and a[r][col]==1:
                a[r]=[a[r][c]^a[row][c] for c in range(n)]
                inv[r]=[inv[r][c]^inv[row][c] for c in range(n)]
        rank+=1; row+=1; col+=1
    return rank,(inv if rank==n else None)

def random_invertible_matrix_and_c(rbytes: List[int]):
    idx=0
    for _ in range(64):
        A=[[0]*8 for _ in range(8)]
        for col in range(8):
            b=rbytes[(idx+col)%len(rbytes)]
            for row in range(8):
                A[row][col]=(b>>(7-row))&1
        cbyte=rbytes[(idx+8)%len(rbytes)]
        c=[(cbyte>>(7-i))&1 for i in range(8)]
        rank,_=gf2_rank_and_inverse(A)
        if rank==8: return A,c
        idx=(idx+9)%len(rbytes)
    raise ValueError("No se pudo derivar A invertible con los bytes dados.")

def gf2_matvec_mul(A: List[List[int]], x_byte: int) -> int:
    xb=[(x_byte>>(7-i))&1 for i in range(8)]
    y=[0]*8
    for i in range(8):
        acc=0
        for j in range(8):
            acc^=(A[i][j]&xb[j])
        y[i]=acc
    out=0
    for i in range(8):
        out|=(y[i]<<(7-i))
    return out

def gf2_add_byte(a: int, c_bits: List[int]) -> int:
    cv=0
    for i in range(8):
        cv|=(c_bits[i]<<(7-i))
    return a ^ cv


# ==== Métricas criptográficas ====

def ddt_full(S: List[int]):
    D=[[0]*256 for _ in range(256)]
    for dx in range(256):
        for x in range(256):
            dy=S[x]^S[x^dx]; D[dx][dy]+=1
    return D

def du_max(DDT): 
    return max(max(row) for row in DDT[1:])

def avalanche_exact(S: List[int]) -> float:
    def hw(a): return bin(a).count("1")
    tot=0; trials=0
    for x in range(256):
        y=S[x]
        for b in range(8):
            y2=S[x^(1<<b)]
            tot+=hw(y^y2); trials+=1
    return tot/trials

def walsh_f(f_bits: List[int]):
    W=[1-2*fb for fb in f_bits]; n=256; h=1
    while h<n:
        for i in range(0,n,h*2):
            for j in range(i,i+h):
                x=W[j]; y=W[j+h]; W[j]=x+y; W[j+h]=x-y
        h*=2
    return W

def nl_component(f_bits: List[int]):
    W=walsh_f(f_bits)
    m=max(abs(v) for v in W)
    return 128 - (m//2)

def nl_min_bits(S: List[int]):
    vals=[]
    for bit in range(8):
        fb=[(S[x]>>bit)&1 for x in range(256)]
        vals.append(nl_component(fb))
    return min(vals), vals


# ==== Validación de métricas ====

DEFAULT_CRITERIA={
    "require_bijective": True,
    "avalanche_min": 3.8,
    "avalanche_max": 4.2,
    "du_max": 10,
    "nl_min": 102
}

def validate(S: List[int], criteria=DEFAULT_CRITERIA):
    rep = {"bijective": len(set(S)) == 256}

    # Calcular SIEMPRE todas las métricas
    rep["avalanche_mean"] = avalanche_exact(S)
    D = ddt_full(S)
    rep["du_max"] = du_max(D)
    nl_min, nls = nl_min_bits(S)
    rep["nl_min_bit"] = nl_min
    rep["nl_bits"] = nls

    # Evaluar criterios
    if criteria["require_bijective"] and not rep["bijective"]:
        rep["pass"] = False
    elif not (criteria["avalanche_min"] <= rep["avalanche_mean"] <= criteria["avalanche_max"]):
        rep["pass"] = False
    elif rep["du_max"] > criteria["du_max"]:
        rep["pass"] = False
    elif rep["nl_min_bit"] < criteria["nl_min"]:
        rep["pass"] = False
    else:
        rep["pass"] = True

    return rep



# ==== Generación principal ====

def build_sbox_signed(api_key: str):
    P, signed_perm = randomorg_generate_signed_integers(api_key, 256, 0, 255, False, request_id=1)
    ok = randomorg_verify_signature(signed_perm["random"], signed_perm["signature"], request_id=1001)
    if not ok:
        raise SignatureVerificationError("Firma inválida en permutación.")
    if len(set(P))!=256:
        raise RandomOrgError("La permutación no contiene 256 valores únicos.")

    extra, signed_extra = randomorg_generate_signed_integers(api_key, 32, 0, 255, True, request_id=2)
    ok2 = randomorg_verify_signature(signed_extra["random"], signed_extra["signature"], request_id=1002)
    if not ok2:
        raise SignatureVerificationError("Firma inválida en bytes adicionales.")

    A,c = random_invertible_matrix_and_c(extra)

    S=[0]*256
    for x in range(256):
        y = gf2_matvec_mul(A, P[x])
        S[x] = gf2_add_byte(y, c)

    digest = hashlib.sha256(bytes(S)).hexdigest()
    meta = {"perm_signed": signed_perm, "extra_signed": signed_extra}
    return S, digest, A, c, meta


# ==== Programa principal ====

def main():
    ap = argparse.ArgumentParser(description="S-box 8x8 con RANDOM.ORG Signed API + verificación de firma")
    ap.add_argument("--out", required=True, help="Ruta de salida para el artefacto JSON")
    ap.add_argument("--du-max", type=int, default=DEFAULT_CRITERIA["du_max"])
    ap.add_argument("--nl-min", type=int, default=DEFAULT_CRITERIA["nl_min"])
    ap.add_argument("--aval-min", type=float, default=DEFAULT_CRITERIA["avalanche_min"])
    ap.add_argument("--aval-max", type=float, default=DEFAULT_CRITERIA["avalanche_max"])
    args = ap.parse_args()

    api_key = os.environ.get("RANDOM_ORG_API_KEY")
    if not api_key:
        raise SystemExit("RANDOM_ORG_API_KEY no definido. Exporta tu clave antes de ejecutar.")

    criteria = dict(DEFAULT_CRITERIA)
    criteria.update({
        "du_max": args.du_max,
        "nl_min": args.nl_min,
        "avalanche_min": args.aval_min,
        "avalanche_max": args.aval_max
    })

    attempt = 1
    while True:
        print(f"\n🧩 Attempt  #{attempt}: generating S-box with RANDOM.ORG…")
        S, digest, A, c, meta = build_sbox_signed(api_key)
        rep = validate(S, criteria)

        print("Metrics:", json.dumps({
            "bijective": rep.get("bijective"),
            "avalanche_mean": rep.get("avalanche_mean"),
            "du_max": rep.get("du_max"),
            "nonlinearity_min_bit": rep.get("nl_min_bit")
        }, ensure_ascii=False))

        if rep.get("pass"):
            print("✅ Valid S-box found.")
            break
        else:
            print("❌ S-box rechazada. Reintentando...\n")
            attempt += 1
            time.sleep(2)  # evita saturar el límite de la API

    artifact = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "sha256": digest,
        "criteria": criteria,
        "metrics": {
            "bijective": rep.get("bijective"),
            "avalanche_mean": rep.get("avalanche_mean"),
            "du_max": rep.get("du_max"),
            "nonlinearity_min_bit": rep.get("nl_min_bit"),
            "nonlinearity_bits": rep.get("nl_bits")
        },
        "sbox": S,
        "affine": {"A_columns_bitorder_MSB_top": A, "c_bits_MSB_first": c},
        "randomorg_signed_meta": meta,
        "passed": True
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(artifact, f, ensure_ascii=False, indent=2)

    print(f"\n💾 Signed artifact saved to  {args.out}")
    print("SHA-256:", digest)


# ==== Ejecución directa ====

if __name__ == "__main__":
    main()
