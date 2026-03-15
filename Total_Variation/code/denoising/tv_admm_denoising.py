# ノイズを含む画像復元(y=x+v)
# 全変動正則化を用いた最適化問題をADMMで解く

import os
import numpy as np
import cv2
from scipy.fftpack import fft2, ifft2
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity
import matplotlib.pyplot as plt
import time
import requests

# グラフ全体のフォント（Computer Modern）に合わせる設定
plt.rcParams['font.family'] = 'serif'
plt.rcParams['mathtext.fontset'] = 'cm' 

# 保存と通知の設定
DISCORD_WEBHOOK_URL = 'https://discord.com/api/webhooks/1313038750023942204/I_bKSXATJu0IsHH6Fi9tgMXn67KkEMba94WH5XE3SE-bwoImQIEKzTss_zHIqnRFvTew'
# タイマー開始
start_time = time.time()

# 補助関数
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

# Isotropic TV (l1,2ノルム) のための近接作用素
def shrink(v_x, v_y, threshold):
    v_norm = np.sqrt(v_x**2 + v_y**2)
    v_norm[v_norm == 0] = 1.0
    shrinkage_factor = np.maximum(0, 1 - threshold / v_norm)
    z_x = shrinkage_factor * v_x
    z_y = shrinkage_factor * v_y
    return z_x, z_y

# l1,2ノルム球への射影
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

# l2ノルム球への射影
def project_l2_ball(v, y, sigma):
    diff = v - y
    dist = np.linalg.norm(diff)
    return v if dist <= sigma else y + sigma * diff / dist

# Total Variationを計算する関数
def calculate_tv(image):
    # np.gradientは(dy, dx)の順で勾配を返す
    dy, dx = np.gradient(image)
    tv = np.sum(np.sqrt(dx**2 + dy**2))
    return tv

# 各手法のADMM
# No constraint (QP) : min_x 1/2*||x - y||_2^2 + lam * TV(x)
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

    x = np.clip(x, 0, 1)
    
    return x


# TV constraint (LS) : min_x ||x - y||_2^2  subject to  TV(x) <= tau
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

    x = np.clip(x, 0, 1)
    
    return x


# Data fidelity constraint (BP) : min_x TV(x)  subject to  ||x - y||_2 <= sigma^2
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
        
        u1_x += (dx_x - z1_x)
        u1_y += (dy_x - z1_y)
        u2 += (x - z2)
        
    x = np.clip(x, 0, 1)
    
    return x

# メイン処理
if __name__ == '__main__':
    # 基本設定
    IMG_PATH = '/home/user/program/Parameter-sensitivity/Total_Variation/image/cameraman.png'
    # IMG_PATH = '/home/user/program/Parameter-sensitivity/Total_Variation/image/house.png'
    # IMG_PATH = '/home/user/program/Parameter-sensitivity/Total_Variation/image/pepper.png'
    # IMG_PATH = '/home/user/program/Parameter-sensitivity/Total_Variation/image/lax.png'
    SAVE_DIR = '/home/user/program/Parameter-sensitivity/Total_Variation/result_tv_admm_denoising'
    image_filename = os.path.basename(IMG_PATH) # ファイル名を取得
    IMG_SAVE_DIR = os.path.join(SAVE_DIR, 'image')
    os.makedirs(IMG_SAVE_DIR, exist_ok=True)
    FIG_SAVE_DIR = os.path.join(SAVE_DIR, 'figure')
    os.makedirs(FIG_SAVE_DIR, exist_ok=True)
    SUM_SAVE_DIR = os.path.join(SAVE_DIR, 'table')
    os.makedirs(SUM_SAVE_DIR, exist_ok=True)
    
    RHO = 5.0
    EP = 100
    noise_level = 0.05 # ノイズレベル
    
    # 画像の読み込みとノイズ付加
    orig_img = cv2.imread(IMG_PATH, cv2.IMREAD_GRAYSCALE).astype(np.float64) / 255.0
    tv_orig = calculate_tv(orig_img) # 原画像の全変動を計算
    print(f'原画像{image_filename}の全変動：{tv_orig}')
    np.random.seed(0)
    noise = np.random.randn(*orig_img.shape) * noise_level
    noisy_img = np.clip(orig_img + noise, 0, 1)

    # 原画像と劣化画像を保存
    # cv2.imwrite(os.path.join(IMG_SAVE_DIR, 'original_image.png'), (orig_img * 255).astype(np.uint8))
    cv2.imwrite(os.path.join(IMG_SAVE_DIR, f'noise_{noise_level}_{image_filename}'), (noisy_img * 255).astype(np.uint8))

# ################### ノイズレベル 0.1 ################    
#     # 各手法のパラメータ範囲を指定(重要)
#     # No constraint (QP) : λ
#     start_lambda = 0.005
#     end_lambda = 1
#     LAM_array = np.logspace(np.log10(start_lambda), np.log10(end_lambda), num=1000)

#     # TV constraint (LS) : τ
#     start_tau = 150
#     end_tau = 30000
#     TAU_array = np.logspace(np.log10(start_tau), np.log10(end_tau), num=1000)

#     # Data fidelity constraint (BP) : σ
#     start_sigma = 1
#     end_sigma = 300
#     SIGMA_array = np.logspace(np.log10(start_sigma), np.log10(end_sigma), num=1000)
    
    
################### ノイズレベル 0.05 ################    
    # 各手法のパラメータ範囲を指定(重要)
    # No constraint : λ
    start_lambda = 0.002
    end_lambda = 0.5
    LAM_array = np.logspace(np.log10(start_lambda), np.log10(end_lambda), num=50)

    # TV constraint : τ
    tv_orig = np.sum(np.sqrt(np.gradient(orig_img, axis=1)**2 + np.gradient(orig_img, axis=0)**2))
    start_tau = 200
    end_tau = 30000
    TAU_array = np.logspace(np.log10(start_tau), np.log10(end_tau), num=50)

    # Data fidelity constraint : σ
    sigma_noise = np.linalg.norm(noise)
    start_sigma = 1.0
    end_sigma = 150
    SIGMA_array = np.logspace(np.log10(start_sigma), np.log10(end_sigma), num=50)


    # 計算実行
    results = {'qp': {}, 'ls': {}, 'bp': {}}
    solvers = {'qp': tv_admm_qp, 'ls': tv_admm_ls, 'bp': tv_admm_bp}
    params = {'qp': LAM_array, 'ls': TAU_array, 'bp': SIGMA_array}
    
    for method, solver_func in solvers.items():
        print(f"Processing {method.upper()}...")
        for p_val in params[method]:
            restored = solvers[method](noisy_img, p_val, RHO, EP)
            results[method][p_val] = {
                'mse': mean_squared_error(orig_img, restored),
                'psnr': peak_signal_noise_ratio(orig_img, restored, data_range=1.0),
                'ssim': structural_similarity(orig_img, restored, data_range=1.0),
                'tv': calculate_tv(restored), # TV値を計算して保存
                'image': restored
            }
    
    # 最適パラメータの決定と正規化
    final_metrics = {}
    normalized_axes_psnr = {}
    normalized_axes_ssim = {}
    optimal_params = {}
    
    print("\n--- Optimal Parameter Analysis ---")
    for method in ['qp', 'ls', 'bp']:
        final_metrics[method] = {
            'mse': [results[method][p]['mse'] for p in params[method]],
            'psnr': [results[method][p]['psnr'] for p in params[method]],
            'ssim': [results[method][p]['ssim'] for p in params[method]],
            'tv': [results[method][p]['tv'] for p in params[method]]
        }
        
        # 各指標での最適値を計算
        idx_psnr = np.argmax(final_metrics[method]['psnr'])
        idx_ssim = np.argmax(final_metrics[method]['ssim'])
        idx_mse = np.argmin(final_metrics[method]['mse'])
        
        optimal_params[method] = {
            'psnr': params[method][idx_psnr],
            'ssim': params[method][idx_ssim],
            'mse': params[method][idx_mse]
        }
        
        normalized_axes_psnr[method] = params[method] / optimal_params[method]['psnr']
        normalized_axes_ssim[method] = params[method] / optimal_params[method]['ssim']
        print(f"Optimal for {method.upper()}: PSNR-based={optimal_params[method]['psnr']:.4f}, SSIM-based={optimal_params[method]['ssim']:.4f}, MSE-based={optimal_params[method]['mse']:.4f}")

        # 復元画像保存 (normalized param = 1e-1, 1e0, 1e1)
        for scale in [1e-1, 1e0, 1e1]:            
            if method == 'ls':
                target_param = optimal_params[method]['ssim'] * scale
            else:
                target_param = optimal_params[method]['psnr'] * scale
            # 最も近い値を選ぶ
            closest_param = params[method][np.argmin(np.abs(params[method] - target_param))]
            restored_img = results[method][closest_param]['image']
            cv2.imwrite(os.path.join(IMG_SAVE_DIR, f"restored_{method}_noise{noise_level}_np{scale:.0e}_{image_filename}"), (restored_img * 255).astype(np.uint8))

    # 結果のテキストファイル出力
    summary_text = f"ADMM TV Denoising Results Summary for {image_filename}\n" # ファイル名を追加
    summary_text += "="*70 + "\n\n"
    summary_text += f"Original Image TV: {tv_orig:.2f}\n"
    summary_text += f"Noisy Image TV:    {calculate_tv(noisy_img):.2f}\n\n"
    
    for method in ['qp', 'ls', 'bp']:
        summary_text += f"--- {method.upper()} Optimal Parameters ---\n"
        summary_text += f"Based on Min MSE  : {optimal_params[method]['mse']:.6f}\n"
        summary_text += f"Based on Max PSNR : {optimal_params[method]['psnr']:.6f}\n"
        summary_text += f"Based on Max SSIM : {optimal_params[method]['ssim']:.6f}\n\n"
    summary_file_path = os.path.join(SUM_SAVE_DIR, f'optimal_parameters_summary_noise_{noise_level}_{image_filename}.txt')
    with open(summary_file_path, 'w') as f: f.write(summary_text)
    print(f"\nOptimal parameters summary saved to {summary_file_path}")


    # 感度グラフ描画
    metrics_to_plot = {
        'MSE': {'data_key': 'mse', 'plot_func': plt.loglog, 'ylabel': 'Mean Squared Error'},
        'PSNR': {'data_key': 'psnr', 'plot_func': plt.semilogx, 'ylabel': 'PSNR (dB)'},
        'SSIM': {'data_key': 'ssim', 'plot_func': plt.semilogx, 'ylabel': 'SSIM'}
    }
    
    colors = {'qp': 'blue', 'ls': 'green', 'bp': 'orange'}
    labels = {'qp': 'No constraint ($\lambda$)', 'ls': 'TV constraint ($\\tau$)', 'bp': 'Data fidelity constraint ($\sigma$)'}
    linestyles = {'qp': '-', 'ls': '--', 'bp': ':'}

    for metric_name, plot_info in metrics_to_plot.items():
        plt.figure(figsize=(10, 7))
        
        for method in ['qp', 'ls', 'bp']:
            
            if metric_name == 'SSIM':
                x_axis_data = normalized_axes_ssim[method]
            else:
                x_axis_data = normalized_axes_psnr[method]         
            
            plot_info['plot_func'](
                x_axis_data,
                final_metrics[method][plot_info['data_key']],
                color=colors[method],
                label=labels[method],
                linestyle=linestyles[method],
                linewidth=3
            )

        plt.xlabel('Normalized Parameter', fontsize=30)
        plt.ylabel(plot_info['ylabel'], fontsize=30)
        plt.tick_params(labelsize=30)
        plt.legend(fontsize=24)
        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.xlim(1e-1, 1e1)       
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_SAVE_DIR, f'parameter_sensitivity_tv_admm_{metric_name}_noise_{noise_level}_{image_filename}.pdf'))
        plt.savefig(os.path.join(FIG_SAVE_DIR, f'parameter_sensitivity_tv_admm_{metric_name}_noise_{noise_level}_{image_filename}.png'))
        plt.show()
    
# 実行完了通知
end_time = time.time()
hours, rem = divmod(end_time - start_time, 3600)
minutes, seconds = divmod(rem, 60)
exec_time_str = f"実行時間: {int(hours)}時間 {int(minutes)}分 {int(seconds)}秒"
data = {"content": f"\ntv全処理完了!\n{exec_time_str}"}
requests.post(DISCORD_WEBHOOK_URL, data=data) 