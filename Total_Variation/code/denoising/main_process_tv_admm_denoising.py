# main_process.py
# ノイズを含む画像復元(y=x+v) - 計算およびデータ保存編

import os
import json
import numpy as np
import cv2
from scipy.fftpack import fft2, ifft2
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity
import time
import requests

# 保存と通知の設定
DISCORD_WEBHOOK_URL = 'https://discord.com/api/webhooks/1313038750023942204/I_bKSXATJu0IsHH6Fi9tgMXn67KkEMba94WH5XE3SE-bwoImQIEKzTss_zHIqnRFvTew'

# --- 補助関数 ---
def psf2otf(psf, shape):
    psf_shape = psf.shape
    padded_psf = np.zeros(shape)
    padded_psf[:psf_shape[0], :psf_shape[1]] = psf
    for i, s in enumerate(psf_shape):
        padded_psf = np.roll(padded_psf, -s // 2, axis=i)
    return fft2(padded_psf)

def get_difference_operators(shape):
    dx_filter = np.array([[1, -1]])
    dy_filter = np.array([[1], [-1]])
    otf_dx = psf2otf(dx_filter, shape)
    otf_dy = psf2otf(dy_filter, shape)
    return otf_dx, otf_dy

def shrink(v_x, v_y, threshold):
    v_norm = np.sqrt(v_x**2 + v_y**2)
    v_norm[v_norm == 0] = 1.0
    shrinkage_factor = np.maximum(0, 1 - threshold / v_norm)
    z_x = shrinkage_factor * v_x
    z_y = shrinkage_factor * v_y
    return z_x, z_y

def project_l12_ball(v_x, v_y, tau):
    norms = np.sqrt(v_x**2 + v_y**2)
    total_norm = np.sum(norms)
    if total_norm <= tau:
        return v_x, v_y
    v_flat = norms.flatten()
    if np.sum(v_flat) <= tau:
        return v_x, v_y
    u = -np.sort(-v_flat)
    sv = np.cumsum(u)
    rho = np.where(u > (sv - tau) / (np.arange(len(u)) + 1))[0][-1]
    theta = (sv[rho] - tau) / (rho + 1)
    w = np.maximum(v_flat - theta, 0).reshape(norms.shape)
    norms[norms == 0] = 1.0
    scale = w / norms
    return v_x * scale, v_y * scale

def project_l2_ball(v, y, sigma):
    diff = v - y
    dist = np.linalg.norm(diff)
    return v if dist <= sigma else y + sigma * diff / dist

def calculate_tv(image):
    dy, dx = np.gradient(image)
    tv = np.sum(np.sqrt(dx**2 + dy**2))
    return tv

# --- ADMM Solvers ---
def tv_admm_qp(noisy_img, lam, rho, num_iterations):
    img_shape = noisy_img.shape
    x = noisy_img.copy()
    z_x, z_y, u_x, u_y = [np.zeros(img_shape) for _ in range(4)]
    otf_dx, otf_dy = get_difference_operators(img_shape)
    otf_dx_conj, otf_dy_conj = np.conj(otf_dx), np.conj(otf_dy)
    denominator = 1.0 + rho * (otf_dx_conj * otf_dx + otf_dy_conj * otf_dy)
    fft_y = fft2(noisy_img)
    
    for _ in range(num_iterations):
        rhs_x = fft_y + rho * (otf_dx_conj * fft2(z_x - u_x) + otf_dy_conj * fft2(z_y - u_y))
        fft_x = rhs_x / denominator
        x = np.real(ifft2(fft_x))
        dx_x, dy_x = np.real(ifft2(otf_dx * fft_x)), np.real(ifft2(otf_dy * fft_x))
        v_x, v_y = dx_x + u_x, dy_x + u_y
        z_x, z_y = shrink(v_x, v_y, lam / rho)
        u_x += (dx_x - z_x)
        u_y += (dy_x - z_y)
    return np.clip(x, 0, 1)

def tv_admm_ls(noisy_img, tau, rho, num_iterations):
    img_shape = noisy_img.shape
    x = noisy_img.copy()
    z_x, z_y, u_x, u_y = [np.zeros(img_shape) for _ in range(4)]
    otf_dx, otf_dy = get_difference_operators(img_shape)
    otf_dx_conj, otf_dy_conj = np.conj(otf_dx), np.conj(otf_dy)
    denominator = 2.0 + rho * (otf_dx_conj * otf_dx + otf_dy_conj * otf_dy)
    fft_y = fft2(noisy_img)
    
    for _ in range(num_iterations):
        rhs_x = 2 * fft_y + rho * (otf_dx_conj * fft2(z_x - u_x) + otf_dy_conj * fft2(z_y - u_y))
        fft_x = rhs_x / denominator
        x = np.real(ifft2(fft_x))
        dx_x, dy_x = np.real(ifft2(otf_dx * fft_x)), np.real(ifft2(otf_dy * fft_x))
        v_x, v_y = dx_x + u_x, dy_x + u_y
        z_x, z_y = project_l12_ball(v_x, v_y, tau)
        u_x += (dx_x - z_x); u_y += (dy_x - z_y)
    return np.clip(x, 0, 1)

def tv_admm_bp(noisy_img, sigma, rho, num_iterations):
    img_shape = noisy_img.shape
    x = noisy_img.copy()
    z1_x, z1_y, z2 = [np.zeros(img_shape) for _ in range(3)]
    u1_x, u1_y, u2 = [np.zeros(img_shape) for _ in range(3)]
    otf_dx, otf_dy = get_difference_operators(img_shape)
    otf_dx_conj, otf_dy_conj = np.conj(otf_dx), np.conj(otf_dy)
    denominator = (otf_dx_conj * otf_dx + otf_dy_conj * otf_dy) + 1.0
    
    for _ in range(num_iterations):
        rhs_x = otf_dx_conj * fft2(z1_x - u1_x) + otf_dy_conj * fft2(z1_y - u1_y) + fft2(z2 - u2)
        fft_x = rhs_x / denominator
        x = np.real(ifft2(fft_x))
        dx_x, dy_x = np.real(ifft2(otf_dx * fft_x)), np.real(ifft2(otf_dy * fft_x))
        v1_x, v1_y = dx_x + u1_x, dy_x + u1_y
        z1_x, z1_y = shrink(v1_x, v1_y, 1.0 / rho)
        v2 = x + u2
        z2 = project_l2_ball(v2, noisy_img, sigma)
        u1_x += (dx_x - z1_x); u1_y += (dy_x - z1_y)
        u2 += (x - z2)
    return np.clip(x, 0, 1)

# --- メイン処理 ---
if __name__ == '__main__':
    start_time = time.time()
    
    # パス設定
    IMG_PATH = '/home/user/program/Parameter-sensitivity/Total_Variation/image/cameraman.png'
    SAVE_DIR = '/home/user/program/Parameter-sensitivity/Total_Variation/result_tv_admm_denoising'
    image_filename = os.path.basename(IMG_PATH)
    
    IMG_SAVE_DIR = os.path.join(SAVE_DIR, 'image')
    DATA_SAVE_DIR = os.path.join(SAVE_DIR, 'data') # データ保存用ディレクトリ
    os.makedirs(IMG_SAVE_DIR, exist_ok=True)
    os.makedirs(DATA_SAVE_DIR, exist_ok=True)
    
    # パラメータ設定
    RHO = 5.0
    EP = 100
    noise_level = 0.05
    
    # 画像読み込み・ノイズ付加
    orig_img = cv2.imread(IMG_PATH, cv2.IMREAD_GRAYSCALE).astype(np.float64) / 255.0
    tv_orig = calculate_tv(orig_img)
    print(f'Processing {image_filename} | Original TV: {tv_orig:.4f}')
    
    np.random.seed(0)
    noise = np.random.randn(*orig_img.shape) * noise_level
    noisy_img = np.clip(orig_img + noise, 0, 1)
    tv_noisy = calculate_tv(noisy_img)
    cv2.imwrite(os.path.join(IMG_SAVE_DIR, f'noise_{noise_level}_{image_filename}'), (noisy_img * 255).astype(np.uint8))

    # パラメータ範囲の設定
    start_lambda, end_lambda = 0.002, 0.5
    LAM_array = np.logspace(np.log10(start_lambda), np.log10(end_lambda), num=10)

    start_tau, end_tau = 200, 30000
    TAU_array = np.logspace(np.log10(start_tau), np.log10(end_tau), num=10)

    start_sigma, end_sigma = 1.0, 150
    SIGMA_array = np.logspace(np.log10(start_sigma), np.log10(end_sigma), num=10)

    # 計算設定
    solvers = {'qp': tv_admm_qp, 'ls': tv_admm_ls, 'bp': tv_admm_bp}
    params = {'qp': LAM_array, 'ls': TAU_array, 'bp': SIGMA_array}
    
    # 結果格納用辞書 (JSON出力用)
    # Numpy配列はJSON化できないためリストに変換して保存します
    output_data = {
        'meta': {
            'image_filename': image_filename,
            'noise_level': noise_level,
            'tv_orig': tv_orig,
            'tv_noisy': tv_noisy,
            'rho': RHO,
            'ep': EP
        },
        'results': {}
    }
    
    # 最適パラメータ記録用（画像保存のため）
    optimal_params = {}
    temp_restored_images = {'qp': {}, 'ls': {}, 'bp': {}}

    # ループ処理
    for method, solver_func in solvers.items():
        print(f"\n--- Method: {method.upper()} ---")
        
        # データリスト初期化
        method_results = {
            'params': params[method].tolist(),
            'mse': [], 'psnr': [], 'ssim': [], 'tv': []
        }
        
        total_steps = len(params[method])
        
        for idx, p_val in enumerate(params[method]):
            # 進捗表示
            print(f"\rProcessing {idx+1}/{total_steps} (Param: {p_val:.4f})", end="")
            
            restored = solver_func(noisy_img, p_val, RHO, EP)
            
            # 指標計算
            mse_val = mean_squared_error(orig_img, restored)
            psnr_val = peak_signal_noise_ratio(orig_img, restored, data_range=1.0)
            ssim_val = structural_similarity(orig_img, restored, data_range=1.0)
            tv_val = calculate_tv(restored)
            
            method_results['mse'].append(mse_val)
            method_results['psnr'].append(psnr_val)
            method_results['ssim'].append(ssim_val)
            method_results['tv'].append(tv_val)
            
            # 画像はメモリに一時保存（最適値判定後、必要なものだけ保存するか、
            # ここではプロット用に後で抽出するためにキーで保持）
            temp_restored_images[method][p_val] = restored

        # JSON用データ構造に格納
        output_data['results'][method] = method_results
        
        # 最適値の計算
        mse_list = method_results['mse']
        psnr_list = method_results['psnr']
        ssim_list = method_results['ssim']
        
        idx_mse = np.argmin(mse_list)
        idx_psnr = np.argmax(psnr_list)
        idx_ssim = np.argmax(ssim_list)
        
        p_mse = params[method][idx_mse]
        p_psnr = params[method][idx_psnr]
        p_ssim = params[method][idx_ssim]
        
        optimal_params[method] = {'mse': p_mse, 'psnr': p_psnr, 'ssim': p_ssim}
        
        print(f"\nCompleted {method.upper()}. Optimal(PSNR): {p_psnr:.4f}")

        # 画像保存 (正規化パラメータ 1e-1, 1e0, 1e1)
        # LSの場合はSSIM基準、他はPSNR基準（元のコードのロジックを踏襲）
        ref_param = p_ssim if method == 'ls' else p_psnr
        
        for scale in [1e-1, 1e0, 1e1]:
            target = ref_param * scale
            # 最も近いパラメータを探す
            closest_idx = np.argmin(np.abs(params[method] - target))
            closest_param = params[method][closest_idx]
            
            img_to_save = temp_restored_images[method][closest_param]
            save_name = f"restored_{method}_noise{noise_level}_np{scale:.0e}_{image_filename}"
            cv2.imwrite(os.path.join(IMG_SAVE_DIR, save_name), (img_to_save * 255).astype(np.uint8))
            print(f"Saved image: {save_name}")

    # JSONファイルへの書き出し
    json_path = os.path.join(DATA_SAVE_DIR, f'results_noise_{noise_level}_{image_filename}.json')
    with open(json_path, 'w') as f:
        json.dump(output_data, f, indent=4)
    print(f"\nAll data saved to: {json_path}")
    
    # 実行完了通知
    end_time = time.time()
    hours, rem = divmod(end_time - start_time, 3600)
    minutes, seconds = divmod(rem, 60)
    exec_time_str = f"実行時間: {int(hours)}時間 {int(minutes)}分 {int(seconds)}秒"
    
    try:
        data = {"content": f"\n[Data Gen] TV処理完了!\nファイル: {json_path}\n{exec_time_str}"}
        requests.post(DISCORD_WEBHOOK_URL, data=data)
    except Exception as e:
        print(f"Discord notification failed: {e}")