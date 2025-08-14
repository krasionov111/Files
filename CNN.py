# -*- coding: utf-8 -*-
import time
start_time = time.time()
import os
# Разрешаем дубли OpenMP (быстрое решение; безопасно для оффлайн-обучения)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Ограничим потоки, чтобы не «раскалывало» ядра и было стабильнее
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# Во время пакетной генерации датасета отключим GUI-backend
import matplotlib
matplotlib.use("Agg")  # только если ты не хочешь окно, а просто сохраняешь/рендеришь


"""
Траектории пинцетов (рандом везде) -> Венгерский -> 21 шаг -> WGS на каждом шаге ->
голограммы -> построение входов/выходов для CNN (1024×1024) + видео траекторий.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from scipy.ndimage import maximum_filter, gaussian_filter
import imageio
import torch

# ============================ CONFIG ===========================================
x_factor = 1
N          = 1024*x_factor   # базовое (дизайн) разрешение SLM/u-плоскости
N_RUN      = 1024*x_factor   # рабочее разрешение для запуска (поставьте 4096 у себя)
GRID_SIZE  = 45     # целевая решетка GRID_SIZE x GRID_SIZE
PITCH      = 20*x_factor     # шаг решетки (в пикселях N_RUN)
SIGMA_SPOT = 5*x_factor      # ширина гауссовой «мягкой» цели (px) в u-плоскости
SIGMA_PUMP = 0.16*x_factor   # относительная ширина гаусса на SLM (радиус ~ SIGMA_PUMP*N_RUN)
NUM_STEPS  = 2     # => 21 положение/голограмма (0..20)
ALPHA      = 0.5    # степень апдейта весов в weighted GS
PHASE_FIX_AT_FIRST = 18  # заморозка фазы в u-плоскости на первой итерации
ITERS_FIRST = 50          # итераций WGS на первом шаге
ITERS_NEXT  = 15          # итераций WGS на остальных шагах (старт с предыдущего решения)
ROI        = 4*x_factor            # полуразмер окна для метрик на сайтах (2*ROI+1)^2
STEP_SHOW  = 1           # какой шаг визуализировать подробно
VIDEO_FPS  = 6
VIDEO_MP4  = "tweezer_positions.mp4"
VIDEO_GIF  = "tweezer_positions.gif"
RNG_SEED   = 1
# ===============================================================================

# -------------------- Центрированные FFT --------------------------------------
def fft2c(u):
    if isinstance(u, torch.Tensor):
        return torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(u)))
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(u)))


def ifft2c(u):
    if isinstance(u, torch.Tensor):
        return torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(u)))
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(u)))
def wrap2pi(phi): return np.mod(phi, 2*np.pi)

# -------------------- Идеальная целевая решетка -------------------------------
def grid_centers(N_side, grid_size, pitch):
    ctr = N_side // 2
    half = grid_size // 2
    offs = pitch * (half - 0.5)
    centers = []
    for i in range(grid_size):
        for j in range(grid_size):
            r = ctr - offs + i*pitch  # y
            c = ctr - offs + j*pitch  # x
            centers.append((r, c))
    return np.array(centers, dtype=np.float32)  # (M,2), float — для субпикселя

# -------------------- Случайные стартовые позиции по всему полю ---------------
def random_positions_anywhere(N_side, M, margin=0, seed=0):
    """
    Равномерно по [0..N-1] без ограничений. margin=0 -> занимают весь диапазон.
    (Коллизии допускаются; для строгого разнесения используйте Poisson-disk.)
    """
    rng = np.random.default_rng(seed)
    low, high = margin, N_side-1-margin
    y = rng.uniform(low, high, size=M)
    x = rng.uniform(low, high, size=M)
    return np.stack([y, x], axis=1).astype(np.float32)

# -------------------- Построение мягкой цели в u-плоскости --------------------
def build_target_amplitude_from_positions(N_side, positions, weights, sigma_spot):
    """
    A_target(y,x) = сумма нормированных гауссиан с амплитудой sqrt(w_k).
    positions: (M,2) float (y,x).
    """
    A = np.zeros((N_side, N_side), dtype=np.float32)
    rad = int(3*sigma_spot)
    for (r, c), w in zip(positions, weights):
        if w <= 0: 
            continue
        r0 = int(np.floor(r)) - rad
        c0 = int(np.floor(c)) - rad
        r1 = r0 + 2*rad + 1
        c1 = c0 + 2*rad + 1
        rr0, rr1 = max(0, r0), min(N_side, r1)
        cc0, cc1 = max(0, c0), min(N_side, c1)
        if rr0 >= rr1 or cc0 >= cc1:
            continue
        ys = np.arange(rr0, rr1) - r
        xs = np.arange(cc0, cc1) - c
        X, Y = np.meshgrid(xs, ys)
        g = np.exp(-(X**2 + Y**2)/(2*sigma_spot**2)).astype(np.float32)
        g /= (g.max() if g.max()>0 else 1.0)
        A[rr0:rr1, cc0:cc1] += np.sqrt(w) * g
    A /= (A.max() if A.max()>0 else 1.0)
    return A

# -------------------- Метрики по ROI вокруг позиций ---------------------------
def intens_at_positions(I, positions, roi=3):
    N_side = I.shape[0]
    vals = []
    for (r, c) in positions:
        r0 = max(0, int(np.floor(r))-roi)
        c0 = max(0, int(np.floor(c))-roi)
        r1 = min(N_side, r0+2*roi+1)
        c1 = min(N_side, c0+2*roi+1)
        vals.append(I[r0:r1, c0:c1].mean())
    return np.array(vals, dtype=np.float32)

def uniformity_metric(I, positions, roi=3):
    v = intens_at_positions(I, positions, roi=roi)
    m = v.max() if v.size else 1.0
    v = v / (m if m>0 else 1.0)
    vmax, vmin = v.max(), v.min()
    return 1.0 - (vmax - vmin)/(vmax + vmin + 1e-12)

def efficiency_roi(I, positions, roi=3):
    N_side = I.shape[0]
    mask = np.zeros_like(I, dtype=bool)
    for (r, c) in positions:
        r0 = max(0, int(np.floor(r))-roi)
        c0 = max(0, int(np.floor(c))-roi)
        r1 = min(N_side, r0+2*roi+1)
        c1 = min(N_side, c0+2*roi+1)
        mask[r0:r1, c0:c1] = True
    return float(I[mask].sum() / I.sum())

# -------------------- Один WGS-шаг с phase-fix --------------------------------
def wgs_phase_fixed_one_step(A_slm, positions, w_sites, sigma_spot,
                             iters=20, phase_fix_at=10, alpha=0.5, roi=3,
                             U_init=None, psi_ref_init=None):
    """
    Weighted GS + заморозка фазы в u-плоскости после phase_fix_at итераций.
    Возврат: U_out (SLM комплексное поле), psi_ref_out, w_sites_out, логи.
    """
    if U_init is None:
        U = A_slm * np.exp(1j*np.zeros_like(A_slm))
    else:
        U = A_slm * np.exp(1j*np.angle(U_init))
    psi_ref = None if psi_ref_init is None else psi_ref_init.copy()

    I_tgt = np.ones_like(w_sites, dtype=np.float32)
    unif_log, eff_log = [], []

    for it in range(iters):
        G = fft2c(U)
        I = (np.abs(G)**2).astype(np.float32)
        I /= (I.max() if I.max()>0 else 1.0)

        unif_log.append(uniformity_metric(I, positions, roi=roi))
        eff_log.append(efficiency_roi(I, positions, roi=roi))

        # адаптивные веса по измеренной интенсивности
        I_meas = intens_at_positions(I, positions, roi=roi)
        I_meas = np.clip(I_meas, 1e-8, None)
        w_sites = w_sites * (I_tgt / I_meas)**alpha
        w_sites /= (w_sites.max() if w_sites.max()>0 else 1.0)

        # цель в u-плоскости
        A_tgt = build_target_amplitude_from_positions(A_slm.shape[0], positions, w_sites, sigma_spot)

        # фаза в u-плоскости + phase-fix
        psi = np.angle(G)
        if (psi_ref is None) and (it == phase_fix_at):
            psi_ref = psi.copy()
        psi_use = psi_ref if (psi_ref is not None) else psi

        # навязываем модуль и фазу в u-плоскости
        G = A_tgt * np.exp(1j*psi_use)

        # обратный ход в SLM и навязывание амплитуды накачки
        U_back = ifft2c(G)
        U = A_slm * np.exp(1j*np.angle(U_back))

    return U, psi_ref, w_sites, np.array(unif_log), np.array(eff_log)

# -------------------- Детекция пиков (для показа, не обязательно) -------------
def detect_peaks(I_target, min_dist=3, threshold_rel=0.1):
    """
    Простейший поиск локальных максимумов в целевой интенсивности (u-плоскость).
    """
    It = gaussian_filter(I_target, sigma=1.0)
    neigh = maximum_filter(It, size=2*min_dist+1, mode='nearest')
    peaks_mask = (It == neigh)
    peaks_mask &= (It >= threshold_rel * It.max())
    ys, xs = np.nonzero(peaks_mask)
    return np.stack([ys.astype(np.float32), xs.astype(np.float32)], axis=1)

# -------------------- Билинейное кодирование в 1024×1024 ----------------------
def bilinear_splat_positions(pos_xy_float, out_h=1024, out_w=1024, src_h=4096, src_w=4096, value_per_point=1.0):
    """
    A_input: «расплескиваем» каждую позицию в 4 соседних пикселя (билинейные веса).
    """
    A = np.zeros((out_h, out_w), dtype=np.float32)
    sy = (out_h - 1) / (src_h - 1)
    sx = (out_w - 1) / (src_w - 1)
    for (y, x) in pos_xy_float:
        uy = y * sy; ux = x * sx
        i = int(np.floor(uy)); j = int(np.floor(ux))
        dy = uy - i; dx = ux - j
        for di in (0, 1):
            for dj in (0, 1):
                ii = i + di; jj = j + dj
                if 0 <= ii < out_h and 0 <= jj < out_w:
                    w = (1 - dy if di==0 else dy) * (1 - dx if dj==0 else dx)
                    A[ii, jj] += value_per_point * w
    return A

def bilinear_sample_phase(phi_map, pos_xy_float):
    """
    Выборка фазы в u-плоскости по субпиксельным координатам через билинейную
    интерполяцию на единичной окружности (интерполируем e^{iφ}, затем arg).
    """
    h, w = phi_map.shape
    phases = []
    for (y, x) in pos_xy_float:
        i = int(np.floor(y)); j = int(np.floor(x))
        dy = y - i; dx = x - j
        acc = 0+0j
        for di in (0, 1):
            for dj in (0, 1):
                ii = np.clip(i+di, 0, h-1); jj = np.clip(j+dj, 0, w-1)
                wgt = (1 - dy if di==0 else dy) * (1 - dx if dj==0 else dx)
                acc += wgt * np.exp(1j * phi_map[ii, jj])
        phases.append(np.angle(acc))
    return np.array(phases, dtype=np.float32)

def accumulate_phi_input(phases_at_points, pos_xy_float, out_h=1024, out_w=1024, src_h=4096, src_w=4096):
    """
    φ_input: суммируем комплексные фазоры с билинейными весами в те же 4 пикселя,
    затем берем arg (где есть вклад).
    """
    acc = np.zeros((out_h, out_w), dtype=np.complex64)
    sy = (out_h - 1) / (src_h - 1)
    sx = (out_w - 1) / (src_w - 1)
    for (y, x), phi in zip(pos_xy_float, phases_at_points):
        uy = y * sy; ux = x * sx
        i = int(np.floor(uy)); j = int(np.floor(ux))
        dy = uy - i; dx = ux - j
        for di in (0, 1):
            for dj in (0, 1):
                ii = i + di; jj = j + dj
                if 0 <= ii < out_h and 0 <= jj < out_w:
                    w = (1 - dy if di==0 else dy) * (1 - dx if dj==0 else dx)
                    acc[ii, jj] += w * np.exp(1j*phi)
    phi_input = np.zeros((out_h, out_w), dtype=np.float32)
    nz = (np.abs(acc) > 1e-12)
    phi_input[nz] = np.angle(acc[nz]).astype(np.float32)
    return phi_input

# ====================== Основной конвейер (на N_RUN) ===========================
def run_all(N_side, grid_size, pitch, sigma_spot, sigma_pump_frac,
            num_steps, iters_first, iters_next, phase_fix_at_first,
            roi, alpha, step_show, seed=0, make_video=False, do_plots=False):

    # 1) Амплитуда на SLM (гаусс), нормировка к 1
    u = np.linspace(-0.5, 0.5, N_side, endpoint=False)
    X, Y = np.meshgrid(u, u)
    A_slm = np.exp(-((X**2 + Y**2)/(2*(sigma_pump_frac**2)))).astype(np.float32)
    A_slm /= (A_slm.max() if A_slm.max()>0 else 1.0)

    # 2) Целевая решетка (идеальные позиции)
    target_pos = grid_centers(N_side, grid_size, pitch)
    M = target_pos.shape[0]

    # 3) Случайные стартовые позиции по всему полю
    init_pos = random_positions_anywhere(N_side, M, margin=0, seed=seed+10)

    # 4) Венгерский алгоритм: init -> target
    C = np.sum((init_pos[:,None,:] - target_pos[None,:,:])**2, axis=2)
    row_ind, col_ind = linear_sum_assignment(C)
    init_assigned   = init_pos[row_ind]
    target_assigned = target_pos[col_ind]

    # 5) Линейные траектории (21 кадр)
    positions_steps = np.stack([
        init_assigned + (target_assigned - init_assigned) * (t/num_steps)
        for t in range(num_steps+1)
    ], axis=0).astype(np.float32)  # (num_steps+1, M, 2) — ключевая переменная!

    video_path = None 
    if make_video:
        # 6) Видео траекторий
        frames = []
        for t in range(num_steps+1):
            fig, ax = plt.subplots(figsize=(5,5), dpi=160)
            ax.set_title(f"Step {t}/{num_steps}: positions")
            ax.set_xlim(0, N_side); ax.set_ylim(0, N_side); ax.invert_yaxis()
            ax.scatter(target_pos[:,1], target_pos[:,0], s=6, marker='x', label='target')
            curr = positions_steps[t]
            ax.scatter(curr[:,1], curr[:,0], s=10, label='current')
            ax.legend(loc='upper right'); ax.grid(True, linewidth=0.3)
            fig.canvas.draw()
            frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            frame = np.ascontiguousarray(frame)
            frames.append(frame)
            plt.close(fig)
        try:
            imageio.mimsave(VIDEO_MP4, frames, fps=VIDEO_FPS, macro_block_size=None)
            video_path = VIDEO_MP4
        except Exception:
            imageio.mimsave(VIDEO_GIF, frames, duration=1.0/VIDEO_FPS)
            video_path = VIDEO_GIF

    # 7) WGS на каждом шаге (с прогревом)
    U_prev = None
    psi_ref_prev = None
    w_sites = np.ones(M, dtype=np.float32)

    # Хранилища
    holograms_phi = []        # φ на SLM (рад) для каждого шага
    image_intens  = []        # интенсивность в u-плоскости (для метрик/показа)
    target_intens = []        # целевая интенсивность в u-плоскости (A_tgt^2)
    coords_detected = []      # найденные пики (для демонстрации)

    A_inputs   = []           # 1024×1024 — A_input
    Phi_inputs = []           # 1024×1024 — φ_input
    A_labels   = []           # 1024×1024 — A_label
    Phi_labels = []           # 1024×1024 — φ_label

    # центральное окно 1024×1024 в плоскости SLM
    c0 = N_side//2 - 512
    c1 = c0 + 1024

    uniformities, efficiencies = [], []

    for t in range(num_steps+1):
        pos_t = positions_steps[t]
        iters = iters_first if t == 0 else iters_next              
        phase_fix_at = phase_fix_at_first  if t == 0 else min(phase_fix_at_first , max(1, iters-2))

        U_out, psi_ref_out, w_sites, ulog, elog = wgs_phase_fixed_one_step(
            A_slm=A_slm, positions=pos_t, w_sites=w_sites, sigma_spot=sigma_spot,
            iters=iters, phase_fix_at=phase_fix_at, alpha=alpha, roi=roi,
            U_init=U_prev, psi_ref_init=psi_ref_prev
        )
        U_prev = U_out; psi_ref_prev = psi_ref_out

        # u-плоскость: поле, интенсивность, метрики
        G = fft2c(U_out)
        I = (np.abs(G)**2).astype(np.float32)
        I /= (I.max() if I.max()>0 else 1.0)

        uniformities.append(uniformity_metric(I, pos_t, roi=roi))
        efficiencies.append(efficiency_roi(I, pos_t, roi=roi))

        # φ на SLM
        phi_t = wrap2pi(np.angle(U_out)).astype(np.float32)
        holograms_phi.append(phi_t)

        # целевая интенсивность в u-плоскости (для показа/логов)
        A_tgt = build_target_amplitude_from_positions(N_side, pos_t, w_sites, sigma_spot)
        target_intens.append((A_tgt**2).astype(np.float32))
        image_intens.append(I)

        # (A) Пики из целевой карты (демонстрация; реально pos_t уже известны)
        coords = detect_peaks(A_tgt**2, min_dist=max(2,sigma_spot), threshold_rel=0.2)
        coords_detected.append(coords)

        # (B) Входы CNN (1024×1024):
        #  A_input — билинейное расплескивание позиций
        A_in = bilinear_splat_positions(pos_t, out_h=1024, out_w=1024,
                                        src_h=N_side, src_w=N_side, value_per_point=1.0)
        #  φ_input — выборка φ в u-плоскости в позициях + расплескивание фазоров
        phi_u = np.angle(G).astype(np.float32)
        phases_at_pts = bilinear_sample_phase(phi_u, pos_t)
        Phi_in = accumulate_phi_input(phases_at_pts, pos_t, out_h=1024, out_w=1024,
                                      src_h=N_side, src_w=N_side)

        # (C) Лейблы (1024×1024) из центрального SLM-окна -> FFT
        phi_crop   = phi_t[c0:c1, c0:c1]
        A_slm_crop = A_slm[c0:c1, c0:c1]
        U_crop = A_slm_crop * np.exp(1j*phi_crop)
        G_lab = fft2c(U_crop)
        A_lab = np.abs(G_lab).astype(np.float32)
        A_lab /= (A_lab.max() if A_lab.max()>0 else 1.0)
        
        Phi_lab = np.angle(G_lab).astype(np.float32)

        A_inputs.append(A_in.astype(np.float32))
        Phi_inputs.append(Phi_in.astype(np.float32))
        A_labels.append(A_lab)
        Phi_labels.append(Phi_lab)

    # --- Диагностика для STEP_SHOW ---
    t = int(np.clip(step_show, 0, num_steps))
    pos_t   = positions_steps[t]
    phi_t   = holograms_phi[t]
    I_t     = image_intens[t]
    A_tgt_t = target_intens[t]
    A_in_t  = A_inputs[t]
    Phi_in_t= Phi_inputs[t]
    A_lab_t = A_labels[t]
    Phi_lab_t=Phi_labels[t]

    if do_plots:
        # (1) Что подаем на SLM
        plt.figure(figsize=(10,4))
        plt.subplot(1,2,1); plt.imshow(A_slm); plt.title("SLM amplitude (pump)"); plt.colorbar()
        plt.subplot(1,2,2); plt.imshow(phi_t);  plt.title(f"SLM phase (step {t})"); plt.colorbar()
    
        # (2) FFT от голограммы (u-плоскость)
        G_t  = fft2c(A_slm * np.exp(1j*phi_t))
        It   = (np.abs(G_t)**2).astype(np.float32); It /= (It.max() if It.max()>0 else 1.0)
        Phit = np.angle(G_t).astype(np.float32)
        plt.figure(figsize=(10,4))
        plt.subplot(1,2,1); plt.imshow(It);   plt.title("FFT(hologram): intensity"); plt.colorbar()
        plt.subplot(1,2,2); plt.imshow(Phit); plt.title("FFT(hologram): phase");     plt.colorbar()
    
        # (3) Как выделены координаты из целевой интенсивности
        coords_show = coords_detected[t]
        plt.figure(figsize=(5,5)); plt.imshow(A_tgt_t); plt.title("Target intensity (u-plane)"); plt.colorbar()
        if coords_show.size: plt.scatter(coords_show[:,1], coords_show[:,0], s=5)
    
        # (4) Билинейная интерполяция в 1024×1024 (A_input, φ_input)
        plt.figure(figsize=(10,4))
        plt.subplot(1,2,1); plt.imshow(A_in_t);  plt.title("A_input (1024×1024)");  plt.colorbar()
        plt.subplot(1,2,2); plt.imshow(Phi_in_t);plt.title("phi_input (1024×1024)");plt.colorbar()
    
        # (5) Лейблы из центрального окна SLM -> FFT
        plt.figure(figsize=(10,4))
        plt.subplot(1,2,1); plt.imshow(A_lab_t);  plt.title("A_label (FFT of SLM crop)"); plt.colorbar()
        plt.subplot(1,2,2); plt.imshow(Phi_lab_t);plt.title("phi_label (FFT of SLM crop)");plt.colorbar()

        # (6) Логи по шагам
        uniformities = np.array([uniformity_metric(img, positions_steps[k], roi=ROI) for k, img in enumerate(image_intens)], dtype=np.float32)
        efficiencies = np.array([efficiency_roi(img, positions_steps[k], roi=ROI)    for k, img in enumerate(image_intens)], dtype=np.float32)
        plt.figure(); plt.plot(uniformities); plt.ylim(0,1); plt.xlabel("step"); plt.ylabel("uniformity")
        plt.figure(); plt.plot(efficiencies);           plt.xlabel("step"); plt.ylabel("efficiency (ROI sum / total)")

    # ----------------- Имена итоговых переменных (по просьбе) -------------------
    # целевые интенсивности лежат в переменной:
    # TARGET_INTENSITIES_PER_STEP = target_intens                # list of (N_RUN×N_RUN) float32
    # начальная интенсивность лазера накачки (большой гаусс) — в переменной:
    # PUMP_AMPLITUDE_ON_SLM = A_slm                              # (N_RUN×N_RUN) float32 (амплитуда)
    # голограммы (фаза на SLM, рад) записаны в:
    # HOLOGRAMS_PHASE_PER_STEP = holograms_phi                   # list of (N_RUN×N_RUN) float32
    # координаты пинцетов, извлечённые из целевой интенсивности (для показа):
    # DETECTED_COORDS_PER_STEP = coords_detected                 # list of arrays (K_t×2) [y,x]
    # входы CNN 1024×1024:
    # A_INPUTS_1024  = A_inputs                                  # list of 1024×1024 float32
    # PHI_INPUTS_1024= Phi_inputs                                # list of 1024×1024 float32 [rad]
    # лейблы из FFT центрального окна SLM 1024×1024:
    # A_LABELS_1024  = A_labels                                  # list of 1024×1024 float32
    # PHI_LABELS_1024= Phi_labels                                # list of 1024×1024 float32 [rad]

    return {
        "video_path": video_path,
        "positions_steps": positions_steps,
        "holograms_count": len(holograms_phi),
        "demo_step": int(step_show),
        # данные для датасета:
        "A_inputs":   A_inputs,
        "Phi_inputs": Phi_inputs,
        "A_labels":   A_labels,
        "Phi_labels": Phi_labels,
        # полезное доп.:
        "A_slm": A_slm,
        "target_intens": target_intens,
        "coords_detected": coords_detected,
        "shapes": {
            "SLM_amp": A_slm.shape,
            "phi_SLM": holograms_phi[0].shape if holograms_phi else None,
            "A_input": A_inputs[0].shape if A_inputs else None,
            "phi_input": Phi_inputs[0].shape if Phi_inputs else None,
            "A_label": A_labels[0].shape if A_labels else None,
            "phi_label": Phi_labels[0].shape if Phi_labels else None,
        }
    }















"""
CNN 'как в статье' для генерации амплитуды/фазы в u-плоскости из (A_in, phi_in).
- Подготовка датасета: N независимых прогонов твоего симулятора (каждый -> 21 шаг).
- Архитектура: 3 входные свёртки -> 3 residual-блока -> 1 выходная свёртка (все 3×3, 16 каналов),
  вход 2 канала (A_in, phi_in), выход 2 канала (A_pred, phi_pred) — точно по SI.
- Лоссы: амплитуда L1 с ROI-весами (усиление около пинцетов) + слабый global=0.001; фаза L2 по завернутой разности
  + 0.1 * L2 по незавёрнутой; общий баланс: амплитудный лосс умножается ×10 (как в SI).
- Метрики: амплитуда MAE (ROI и global), фаза RMSE по завернутой разности (ROI и global).
- Тренировочный цикл с прогрессом и сохранением лучшей модели.

Ссылки на то, что повторяем:
- Подготовка входов/лейблов 8192->1024 (A_in/phi_in; FFT центрального окна) — SI 4.1 Data preparation.
- Архитектура (2->16, 16->16, 16->16; 3 residual blocks; выход 16->2) — SI 4.2 Model design.
- Лоссы/веса/балансировку — SI 4.2 Loss function paragraph.

"""

import os, math, time, json
import numpy as np
from glob import glob
from typing import Tuple, Dict

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from glob import glob
import os, re, numpy as np, time
# Если используешь SciPy для морфологии при упаковке датасета:
try:
    from scipy.ndimage import maximum_filter
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False

def next_run_index(data_root):
    files = glob(os.path.join(data_root, "run*.npz"))
    if not files:
        return 0
    # извлекаем номера run из имён run00012_step07.npz
    runs = []
    for f in files:
        m = re.search(r"run(\d+)_step", os.path.basename(f))
        if m:
            runs.append(int(m.group(1)))
    return max(runs)+1 if runs else 0
# ======================================================================================
#                                КОНФИГ
# ======================================================================================
FORCE_REGENERATE = False     # True -> всегда заново
APPEND_MODE      = True      # True -> добавляем новые run'ы к существующим
# Где сохранять/читать .npz с примерами (по 1 файлу на шаг):
DATA_ROOT = "./ai_tweezers_ds_1024"
os.makedirs(DATA_ROOT, exist_ok=True)

# Сколько независимых прогонов симулятора выполнить (каждый даёт 21 шага):
NUM_RUNS_TO_GENERATE = 1         # для быстрого теста; доведи до 5000 для качества как в статье

# SEED0 = int(np.random.randint(0, np.iinfo(np.int32).max))

# Параметры ROI-весов (амплитуда): берем порог по A_in и расширяем окрестность
A_IN_THRESHOLD = 0.03              # порог "там есть пинцет" (после билинейного кодирования)
ROI_DILATE_RADIUS = 3              # радиус расширения (в пикселях 1024×1024)
WEIGHT_ROI = 1.0                   # вес ROI (локальные ошибки)
WEIGHT_GLOBAL = 0.001              # вес глобальной амплитуды (как в SI)

# Баланс лоссов (как в SI)
AMP_SCALE = 10.0                   # множитель амплитудного лосса
PHASE_AUX_COEF = 0.1               # коэффициент для незавёрнутой L2 по фазе

# Тренировка
BATCH_SIZE = 2                      # 1024×1024: держи небольшим, иначе OOM
LR = 2e-4
WEIGHT_DECAY = 1e-4
EPOCHS = 40
NUM_WORKERS = 2
AMP_MIXED = True                    # автокастинг (ускорит/снизит память)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cuda":
    torch.backends.cudnn.benchmark = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

# ======================================================================================
#                   ГЕНЕРАЦИЯ ДАТАСЕТА (из твоего симулятора)
# ======================================================================================


def save_sample_npz(path: str, A_in: np.ndarray, Phi_in: np.ndarray,
                    A_lab: np.ndarray, Phi_lab: np.ndarray, meta: Dict):
    """
    Сохраняем один пример на диск. Все карты размера 1024×1024 float32.
    meta может включать seed, step, индексы и т.п.
    """
    # ROI-маска для амплитуды: порог по A_in и расширение
    if _HAVE_SCIPY:
        roi_mask = (A_in > A_IN_THRESHOLD).astype(np.uint8)
        if ROI_DILATE_RADIUS > 0:
            k = 2 * ROI_DILATE_RADIUS + 1
            roi_mask = maximum_filter(roi_mask, size=k, mode='nearest')
    else:
        # Fallback без SciPy: простое пороговое без расширения
        roi_mask = (A_in > A_IN_THRESHOLD).astype(np.uint8)

    np.savez_compressed(
        path,
        A_in=A_in.astype(np.float32),
        Phi_in=Phi_in.astype(np.float32),
        A_lab=A_lab.astype(np.float32),
        Phi_lab=Phi_lab.astype(np.float32),
        roi=roi_mask.astype(np.uint8),
        meta=json.dumps(meta),
    )

def build_dataset_from_simulator(n_runs: int, start_seed: int = 0, start_run: int = 0):
    """
    Вызывает твою функцию генерации для каждого прогона, забирает списки
    A_inputs/Phi_inputs/A_labels/Phi_labels и сохраняет в .npz файлы.
    Привязка к именам переменных — в твоём "Combined WGS..." уже совпадает.
    """
    # Импортируем функции из твоего симулятора, если они лежат в отдельном модуле.
    # Здесь предполагаем, что тот файл уже исполнен или функции доступны.
    from importlib import import_module
    # Если у тебя функция run_all доступна в текущем пространстве — можно убрать импорт.
    # Иначе, например: sim = import_module("your_simulator_module"); run_all = sim.run_all

    saved = 0
    for k in range(n_runs):
        run_id = start_run + k
        seed = int(start_seed + run_id)
        t0 = time.time()

        # ВАЖНО: здесь вызываем твою run_all(...) С ТЕМИ ЖЕ НАСТРОЙКАМИ, что использовал при подготовке
        # В примере ниже предполагается, что ты уже запускал код и у тебя есть функция run_all в памяти.
        info = run_all(  # type: ignore[name-defined]
            N_side=1024,              # для отладки считаем в 1024×1024
            grid_size=45,
            pitch=20,
            sigma_spot=5,
            sigma_pump_frac=0.16,
            num_steps=20,             # 21 положение
            iters_first=30,
            iters_next=10,
            phase_fix_at_first=18,
            roi=4,
            alpha=0.5,
            step_show=10,
            seed=seed, make_video=False, do_plots=False,
        )
        # В твоём run_all возвращаются списки A_INPUTS_1024 / PHI_INPUTS_1024 / A_LABELS_1024 / PHI_LABELS_1024
        # через замыкание; для простоты извлечём их из глобального пространства, если run_all так реализован.
        # Либо модифицируй run_all так, чтобы он возвращал их явно.
        # В нашем текстовом файле "Combined WGS..." они уже возвращаются в словаре info["shapes"], а сами списки
        # лежат в именованных переменных верхнего уровня — подхватим их:
        A_inputs   = info["A_inputs"]
        Phi_inputs = info["Phi_inputs"]
        A_labels   = info["A_labels"]
        Phi_labels = info["Phi_labels"]
        assert len(A_inputs) == len(Phi_inputs) == len(A_labels) == len(Phi_labels) > 0, \
            "Пустые списки входов/лейблов — проверь, что run_all формирует их как в примере."

        for step_idx, (A_in, Phi_in, A_lab, Phi_lab) in enumerate(zip(A_inputs, Phi_inputs, A_labels, Phi_labels)):
            fn = f"run{run_id:05d}_step{step_idx:02d}.npz"
            save_sample_npz( os.path.join(DATA_ROOT, fn),
                A_in, Phi_in, A_lab, Phi_lab,
                meta=dict(run=run_id, step=step_idx, seed=seed)
            )
            saved += 1

        dt = time.time() - t0
        print(f"[gen] run {run_id+1}/{n_runs} -> {len(A_inputs)} примеров (всего {saved}); {dt:.1f}s")

    print(f"[gen] Готово. Всего сэмплов: {saved}. Файлы в: {DATA_ROOT}")

# ======================================================================================
#                         DATASET / DATALOADER
# ======================================================================================

class TweezersNPZ(Dataset):
    """
    Читает .npz-файлы; возвращает тензоры:
    X: (B, 2, H, W)  -> [A_in, phi_in]
    Y: (B, 2, H, W)  -> [A_label, phi_label]
    W: (B, 1, H, W)  -> ROI-маска для амплитудного лосса
    """
    def __init__(self, files):
        self.files = files

    def __len__(self): return len(self.files)

    def __getitem__(self, i):
        z = np.load(self.files[i])
        A_in   = z["A_in"].astype(np.float32)
        Phi_in = z["Phi_in"].astype(np.float32)
        A_lab  = z["A_lab"].astype(np.float32)
        Phi_lab= z["Phi_lab"].astype(np.float32)
        roi    = z["roi"].astype(np.float32)  # 0/1

        X = np.stack([A_in, Phi_in], axis=0)            # (2,H,W)
        Y = np.stack([A_lab, Phi_lab], axis=0)          # (2,H,W)
        W = roi[None, ...]                               # (1,H,W)
        return torch.from_numpy(X), torch.from_numpy(Y), torch.from_numpy(W)

def split_train_val_test():
    files = sorted(glob(os.path.join(DATA_ROOT, "*.npz")))
    assert files, "Нет данных. Сначала сгенерируй их (build_dataset_from_simulator)."
    # Разделяем по 'run' (не по шагам), чтобы исключить утечку
    runs = sorted(set(int(os.path.basename(f).split("_")[0][3:]) for f in files))
    # 80/10/10 по run'ам
    n = len(runs)
    n_tr = max(1, int(0.8*n))
    n_va = max(1, int(0.1*n))
    run_tr = set(runs[:n_tr])
    run_va = set(runs[n_tr:n_tr+n_va])
    run_te = set(runs[n_tr+n_va:])

    def mask(files, runset):
        out = []
        for f in files:
            run_id = int(os.path.basename(f).split("_")[0][3:])
            if run_id in runset:
                out.append(f)
        return out

    tr_files = mask(files, run_tr)
    va_files = mask(files, run_va)
    te_files = mask(files, run_te)
    print(f"[split] runs: total={len(runs)}; train={len(run_tr)}, val={len(run_va)}, test={len(run_te)}")
    print(f"[split] files: train={len(tr_files)}, val={len(va_files)}, test={len(te_files)}")
    return tr_files, va_files, te_files

# ======================================================================================
#                           МОДЕЛЬ (ровно как в SI)
# ======================================================================================

class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, k=3):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, padding=k//2, bias=False)
        self.bn   = nn.BatchNorm2d(out_ch)
        self.act  = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class ResidualBlock(nn.Module):
    """ Две свёртки 3×3 с BN и ReLU; вход/выход 16 каналов; sum skip. """
    def __init__(self, ch=16):
        super().__init__()
        self.c1 = ConvBNReLU(ch, ch, 3)
        # в оригинале после второй свёртки activation перед суммой может не стоять — оставим как обычно:
        self.c2 = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
        )
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        y = self.c1(x)
        y = self.c2(y)
        return self.act(x + y)

class CNN_AIModule(nn.Module):
    """
    3 входные свёртки (2->16, 16->16, 16->16),
    затем 3 residual блока (16 каналов),
    затем выходная свёртка 16->2.
    """
    def __init__(self):
        super().__init__()
        self.in1 = ConvBNReLU(2, 16, 3)   # входные каналы: [A_in, phi_in]
        self.in2 = ConvBNReLU(16, 16, 3)
        self.in3 = ConvBNReLU(16, 16, 3)
        self.rb1 = ResidualBlock(16)
        self.rb2 = ResidualBlock(16)
        self.rb3 = ResidualBlock(16)
        self.out = nn.Conv2d(16, 2, 3, padding=1)  # выходные каналы: [A_pred, phi_pred]
    def forward(self, x):
        y = self.in3(self.in2(self.in1(x)))
        y = self.rb3(self.rb2(self.rb1(y)))
        y = self.out(y)
        # Никакой активации на выходе: амплитуда и фаза — «сырые»; амплитуду нормируем в лоссе
        return y

# ======================================================================================
#                     ЛОССЫ И МЕТРИКИ (как в статье)
# ======================================================================================

def wrap_to_pi(delta_phi: torch.Tensor) -> torch.Tensor:
    """Заворачиваем разность фаз в (-pi, pi]."""
    return (delta_phi + math.pi) % (2*math.pi) - math.pi

def loss_components(pred: torch.Tensor, gt: torch.Tensor, roi: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
    """
    pred: (B,2,H,W) -> [A_pred, phi_pred]
    gt:   (B,2,H,W) -> [A_lab,  phi_lab]
    roi:  (B,1,H,W) -> ROI-маска (0/1)
    Возвращает: (общий лосс, словарь метрик)
    """
    A_pred, phi_pred = pred[:,0:1], pred[:,1:2]
    A_lab,  phi_lab  = gt[:,0:1],  gt[:,1:2]

    # --- Амплитуда: L1 с ROI весами + слабый глобальный (0.001), масштаб ×10
    # Нормируем A_pred в [0,1] мягко: клип по ReLU6/деление на 6 можно не делать; лучше просто clamp:
    A_pred_n = torch.clamp(A_pred, 0.0, 1.0)

    # ROI (локальный) вклад
    roi = roi.float()
    roi_sum = roi.sum().clamp_min(1.0)
    l1_roi = (roi * torch.abs(A_pred_n - A_lab)).sum() / roi_sum

    # Глобальный слабый вклад
    l1_global = torch.mean(torch.abs(A_pred_n - A_lab))

    amp_loss = AMP_SCALE * (WEIGHT_ROI * l1_roi + WEIGHT_GLOBAL * l1_global)

    # --- Фаза: L2 по завернутой разности + 0.1 * L2 по незавёрнутой
    dphi = phi_pred - phi_lab
    dphi_wrapped = wrap_to_pi(dphi)

    phase_l2 = torch.mean(dphi_wrapped**2)
    phase_l2_aux = torch.mean(dphi**2)  # незавёрнутый

    phase_loss = phase_l2 + PHASE_AUX_COEF * phase_l2_aux

    total = amp_loss + phase_loss

    # --- Метрики для логов
    with torch.no_grad():
        mae_amp_global = torch.mean(torch.abs(A_pred_n - A_lab)).item()
        mae_amp_roi    = (roi * torch.abs(A_pred_n - A_lab)).sum().item() / roi_sum.item()
        rmse_phi_global= torch.sqrt(torch.mean(dphi_wrapped**2)).item()
        rmse_phi_roi   = torch.sqrt(((roi * dphi_wrapped**2).sum() / roi_sum)).item()

    metrics = dict(
        amp_loss=float(amp_loss.item()),
        phase_loss=float(phase_loss.item()),
        mae_amp_global=mae_amp_global,
        mae_amp_roi=mae_amp_roi,
        rmse_phi_global=rmse_phi_global,
        rmse_phi_roi=rmse_phi_roi,
    )
    return total, metrics

# ======================================================================================
#                         ОБУЧЕНИЕ / ВАЛИДАЦИЯ
# ======================================================================================

def train_one_epoch(model, loader, opt, scaler=None):
    model.train()
    t0 = time.time()
    running = []
    for it, (X, Y, W) in enumerate(loader, 1):
        X = X.to(DEVICE, non_blocking=True)
        Y = Y.to(DEVICE, non_blocking=True)
        W = W.to(DEVICE, non_blocking=True)

        opt.zero_grad(set_to_none=True)
        if scaler is not None:
            with torch.autocast(device_type=DEVICE.type, dtype=torch.float16):
                P = model(X)
                loss, _ = loss_components(P, Y, W)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            P = model(X)
            loss, _ = loss_components(P, Y, W)
            loss.backward()
            opt.step()

        running.append(loss.item())
        if it % 20 == 0:
            print(f"[train] it {it}/{len(loader)}  loss={np.mean(running):.4f}")
    return np.mean(running), time.time() - t0

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    losses, mets = [], []
    for X, Y, W in loader:
        X = X.to(DEVICE, non_blocking=True)
        Y = Y.to(DEVICE, non_blocking=True)
        W = W.to(DEVICE, non_blocking=True)
        P = model(X)
        loss, m = loss_components(P, Y, W)
        losses.append(loss.item()); mets.append(m)
    # усредняем метрики
    out = {k: float(np.mean([d[k] for d in mets])) for k in mets[0].keys()} if mets else {}
    return float(np.mean(losses)), out

def main():
    # 1) При необходимости — сгенерировать данные (или пропусти, если уже сгенерил)
    need_generate = FORCE_REGENERATE or (not glob(os.path.join(DATA_ROOT, "*.npz")))
    if need_generate:
        if FORCE_REGENERATE:
            # подчистить каталог
            for f in glob(os.path.join(DATA_ROOT, "*.npz")):
                os.remove(f)
        # стартовый run-индекс
        start_run = 0 if not APPEND_MODE else next_run_index(DATA_ROOT)

        # сид: каждый запуск — новый базовый
        SEED0 = int(np.random.randint(0, np.iinfo(np.int32).max))
        build_dataset_from_simulator(NUM_RUNS_TO_GENERATE, start_seed=SEED0, start_run=start_run)

    # 2) Разделение на train/val/test по run'ам
    tr_files, va_files, te_files = split_train_val_test()
    train_ds = TweezersNPZ(tr_files)
    val_ds   = TweezersNPZ(va_files)
    test_ds  = TweezersNPZ(te_files)

    train_ld = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=(DEVICE.type == "cuda"), persistent_workers=False)
    val_ld   = DataLoader(val_ds, batch_size=1, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=(DEVICE.type == "cuda"), persistent_workers=False)
    test_ld  = DataLoader(test_ds, batch_size=1, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=(DEVICE.type == "cuda"), persistent_workers=False)

    # 3) Модель + оптимизатор
    RESUME = True                                    # включить продолжение
    RESUME_PATH = os.path.join(DATA_ROOT, "best_model.pt")  # или last_model.pt
    
    model = CNN_AIModule().to(DEVICE)
    if DEVICE.type == "cuda" and hasattr(torch, "compile"):
        model = torch.compile(model)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler('cuda') if (AMP_MIXED and DEVICE.type == 'cuda') else None
    

    start_epoch = 1
    best_val = float("inf")
    if RESUME and os.path.exists(RESUME_PATH):
        ckpt = torch.load(RESUME_PATH, map_location=DEVICE)
        model.load_state_dict(ckpt["state_dict"])
        if "opt_state" in ckpt:
            opt.load_state_dict(ckpt["opt_state"])           # <- важно
        if scaler is not None and "scaler_state" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state"])     # <- для AMP
        if "epoch" in ckpt:
            start_epoch = int(ckpt["epoch"]) + 1
        if "best_val" in ckpt:
            best_val = float(ckpt["best_val"])
        print(f"[resume] Продолжаем с эпохи {start_epoch}, best_val={best_val:.4f}")
    else:
        print("[resume] Нет чекпойнта — старт с нуля")
    
    ckpt_path = os.path.join(DATA_ROOT, "best_model.pt")
    # --- цикл обучения ---
    print("[main] Старт обучения")
    for epoch in range(1, EPOCHS+1):
        tr_loss, tr_dt = train_one_epoch(model, train_ld, opt, scaler)
        if val_ld is not None:
            va_loss, va_m = evaluate(model, val_ld)        
            print(f"[epoch {epoch:03d}] train_loss={tr_loss:.4f} ({tr_dt:.1f}s) | "
                  f"val_loss={va_loss:.4f} | "
                  f"MAE_amp[g]={va_m.get('mae_amp_global',0):.4f}  MAE_amp[ROI]={va_m.get('mae_amp_roi',0):.4f} | "
                  f"RMSE_phi[g]={va_m.get('rmse_phi_global',0):.4f}  RMSE_phi[ROI]={va_m.get('rmse_phi_roi',0):.4f}")
        #    сохраняем лучшую модель по val_loss
            if va_loss < best_val:
                best_val = va_loss
                torch.save({
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "opt_state": opt.state_dict(),
                "scaler_state": (scaler.state_dict() if scaler is not None else None),
                "val_loss": va_loss,
                "val_metrics": va_m,
                "best_val": best_val, }, os.path.join(DATA_ROOT, "best_model.pt"))
                print(f"[ckpt] сохранён лучший чекпойнт -> {ckpt_path}")
        # сохраняем «последнюю» всегда
            torch.save({
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "opt_state": opt.state_dict(),
                "scaler_state": (scaler.state_dict() if scaler is not None else None),
                "best_val": best_val,
            }, os.path.join(DATA_ROOT, "last_model.pt"))
    
    # 4) Оценка на тесте
    print("[main] Тестирование лучшей модели...")
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model = CNN_AIModule().to(DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    te_loss, te_m = evaluate(model, test_ld)
    print(f"[test] loss={te_loss:.4f} | "
          f"MAE_amp[g]={te_m.get('mae_amp_global',0):.4f}  MAE_amp[ROI]={te_m.get('mae_amp_roi',0):.4f} | "
          f"RMSE_phi[g]={te_m.get('rmse_phi_global',0):.4f}  RMSE_phi[ROI]={te_m.get('rmse_phi_roi',0):.4f}")

if __name__ == "__main__":
    main()

end_time = time.time()
elapsed_time = end_time - start_time
print("Время выполнения:", time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
