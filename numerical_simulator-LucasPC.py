import numpy as np
import matplotlib.pyplot as plt

# ======================================================================
# 1. PARÂMETROS GERAIS E DA GEOMETRIA 
# ======================================================================
Lambda = 304e-9     
N = 658             
DC = 0.5            
L_total = N * Lambda 

w_c = 500e-9  
w = 640e-9    

H = 1
alpha_param = 1
beta = 1.57
S = 0

T_max = 0.630133
lambda_0 = 1541.5e-9 

# ======================================================================
# 2. ÂNCORAS FÍSICAS (Tabela Lookup do FDE e Calibração 3D)
# ======================================================================
w_data = np.array([400, 410, 420, 430, 440, 450, 460, 470, 480, 490, 500, 510, 520, 530, 540, 550, 560, 570, 580, 590, 600, 610, 620, 630, 640, 650, 660, 670, 680, 690, 700, 710, 720, 730, 740, 750, 760, 770, 780, 790, 800])
neff_data = np.array([2.23788, 2.26416, 2.29488, 2.30995, 2.32644, 2.34504, 2.36652, 2.39222, 2.4261, 2.44085, 2.45518, 2.46907, 2.48243, 2.49532, 2.51053, 2.51755, 2.52537, 2.53437, 2.54505, 2.55822, 2.57623, 2.58398, 2.5916, 2.59906, 2.60632, 2.61341, 2.62186, 2.62566, 2.62994, 2.63492, 2.64093, 2.64849, 2.65907, 2.66361, 2.66811, 2.67255, 2.67691, 2.68119, 2.68635, 2.68864, 2.69123])
ng_data = np.array([4.44573, 4.42866, 4.3999, 4.37245, 4.34524, 4.31917, 4.29612, 4.27922, 4.27222, 4.25636, 4.2405, 4.2249, 4.20981, 4.19523, 4.17493, 4.16047, 4.14568, 4.13086, 4.11683, 4.10514, 4.09797, 4.08853, 4.07914, 4.06992, 4.06098, 4.05231, 4.04037, 4.03245, 4.02421, 4.01575, 4.00749, 4.00027, 3.99541, 3.98971, 3.98402, 3.9784, 3.9729, 3.96754, 3.96012, 3.95536, 3.95035])

poly_neff = np.poly1d(np.polyfit(w_data, neff_data, 3))
poly_ng   = np.poly1d(np.polyfit(w_data, ng_data, 3))

neff1_static = poly_neff(w_c * 1e9)
neff2_static = poly_neff(w * 1e9)
ng1_static   = poly_ng(w_c * 1e9)
ng2_static   = poly_ng(w * 1e9)

L_calibracao = 100 * Lambda 

# CORREÇÃO CRÍTICA: Divisão por L_calibracao para o kappa_max
kappa_max_teo = np.arctanh(np.sqrt(T_max)) / L_calibracao

# ======================================================================
# PASSO 1: GERADOR DO PERFIL ESPACIAL
# ======================================================================
ds_max = Lambda * DC
delta_s_vector = np.zeros(N)
y_raw = np.zeros(N)

for n in range(N):
    x = (2.0 * (n + 1) - N) / N 
    sum_val = 0
    for k in range(1, H + 1):
        Ak = np.sin(k * beta) / (k ** alpha_param)
        sum_val += Ak * np.cos(k * np.pi * x / 2.0)
    y_raw[n] = sum_val

y_min = np.min(y_raw)
y_max = np.max(y_raw)

for n in range(N):
    val_norm = (y_raw[n] - y_min) / (y_max - y_min) if y_max > y_min else 1.0
    if S > 1: 
        levels = S - 1
        val_norm = np.round(val_norm * levels) / levels
        
    delta_s_vector[n] = (1 - val_norm) * ds_max

# ======================================================================
# PASSO 2: MAPEAMENTO DO ACOPLAMENTO 
# ======================================================================
kappa_vector = kappa_max_teo * np.cos(np.pi * delta_s_vector / Lambda)

# ======================================================================
# PASSO 3: LAÇO TMM (Com Calibração Geométrica 3D)
# ======================================================================
wl_start = 1520e-9
wl_stop = 1580e-9
points = 500
wavelengths = np.linspace(wl_start, wl_stop, points)
R_spectrum = np.zeros(points) 

for idx, wl in enumerate(wavelengths):
    neff1_real = neff1_static - ((ng1_static - neff1_static) / lambda_0) * (wl - lambda_0)
    neff2_real = neff2_static - ((ng2_static - neff2_static) / lambda_0) * (wl - lambda_0)
    
    neff_avg = (1 - DC) * neff1_real + DC * neff2_real
    
    # CALIBRAÇÃO 3D: Deslocamento do neff_avg para casar com a física de 1548.5 nm do FDTD
    #neff_avg_3D = neff_avg + 0.0164 
    
    delta = (2 * np.pi * neff_avg / wl) - (np.pi / Lambda)
    T_global = np.eye(2, dtype=complex)
    
    for i in range(N):
        k_i = kappa_vector[i]
        gamma = np.sqrt(k_i**2 + delta**2 + 0j) 
        
        cosh_g = np.cosh(gamma * Lambda)
        sinh_g = np.sinh(gamma * Lambda)
        
        T11 = cosh_g + 1j * (delta / gamma) * sinh_g
        T12 = -1j * (k_i / gamma) * sinh_g
        T21 =  1j * (k_i / gamma) * sinh_g
        T22 = cosh_g - 1j * (delta / gamma) * sinh_g
        
        T_i = np.array([[T11, T12], [T21, T22]])
        T_global = np.dot(T_global, T_i)
        
    r_amplitude = T_global[1, 0] / T_global[0, 0]
    R_spectrum[idx] = np.abs(r_amplitude)**2

# ======================================================================
# 4. PLOTAGEM DOS RESULTADOS
# ======================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

z_axis = np.linspace(0, L_total*1e6, N)
ax1.plot(z_axis, kappa_vector/1000, color='darkorange', linewidth=2)
ax1.set_title("Perfil de Acoplamento Espacial ($\kappa_i$)")
ax1.set_xlabel("Comprimento do Guia ($\mu m$)")
ax1.set_ylabel("Força de Acoplamento ($mm^{-1}$)")
ax1.grid(True)

ax2.plot(wavelengths*1e9, R_spectrum, color='purple', linewidth=2)
ax2.set_title("Espectro de Reflexão Verdadeiro ($S_{11}$)")
ax2.set_xlabel("Comprimento de Onda (nm)")
ax2.set_ylabel("Refletividade (Adimensional)")
ax2.set_xlim(1520, 1580)
ax2.grid(True)

plt.tight_layout()
plt.show()

# ======================================================================
# 4.1 CÁLCULO DA BANDA E CENTRO DO FILTRO (3dB)
# ======================================================================
# Encontra o valor máximo de refletividade
R_max = np.max(R_spectrum)

# O ponto de 3dB em escala linear é metade da potência máxima
R_3dB = R_max * 0.5 

# Encontra os índices onde a curva cruza a linha de 3dB
crossings = np.where(np.diff(np.sign(R_spectrum - R_3dB)))[0]

if len(crossings) >= 2:
    # Pegamos o primeiro e o último cruzamento para ignorar oscilações no flat-top
    idx_left = crossings[0]
    idx_right = crossings[-1]
    
    # Interpolação linear para achar o comprimento de onda exato do cruzamento
    wl_left = wavelengths[idx_left] + (wavelengths[idx_left+1] - wavelengths[idx_left]) * ((R_3dB - R_spectrum[idx_left]) / (R_spectrum[idx_left+1] - R_spectrum[idx_left]))
    
    wl_right = wavelengths[idx_right] + (wavelengths[idx_right+1] - wavelengths[idx_right]) * ((R_3dB - R_spectrum[idx_right]) / (R_spectrum[idx_right+1] - R_spectrum[idx_right]))
    
    # Calcula o centro geométrico entre as duas bordas de 3dB
    wl_center = (wl_left + wl_right) / 2
    bw_3dB = wl_right - wl_left
    
    print("\n--- Análise Espectral do Filtro (3dB) ---")
    print(f"Comprimento Central: {wl_center*1e9:.3f} nm")
    print(f"Banda de Passagem (3dB): {bw_3dB*1e9:.3f} nm")
    print(f"Refletividade Máxima: {R_max:.4f}")
    print("-----------------------------------------")
else:
    print("\n[Aviso] Não foi possível detectar bordas claras de 3dB no espectro.")