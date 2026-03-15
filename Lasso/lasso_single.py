# ノイズとぼけを含む画像復元(y=Ax+v)
# Lassoを用いた最適化問題を解く (SSIM評価あり)
# 1回の復元結果

import os
import numpy as np
from scipy.fftpack import fft2, ifft2
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity
import matplotlib.pyplot as plt
import time
import requests

# 保存と通知の設定
DISCORD_WEBHOOK_URL = 'https://discord.com/api/webhooks/1313038750023942204/I_bKSXATJu0IsHH6Fi9tgMXn67KkEMba94WH5XE3SE-bwoImQIEKzTss_zHIqnRFvTew'
# タイマー開始
start_time = time.time()

# 補助関数 (ソルバー)

# 非制約Lasso(QP_λ)をFISTAで解く
# QP_λ : min (1/2)||y - Ax||_2^2 + lambda*||x||_1
def solve_qp_fista(y, A, lambda_val, max_iter=1000, tol=1e-6):
    m, n = A.shape
    x = np.zeros(n)
    x_prev = np.zeros(n)
    t = 1.0
    t_prev = 1.0
    
    L = np.linalg.norm(A.T @ A, 2)
    eta = 1.0 / L

    for i in range(max_iter):
        y_k = x + ((t_prev - 1) / t) * (x - x_prev)
        grad_f = A.T @ (A @ y_k - y)
        z_k = y_k - eta * grad_f
        x_new = np.sign(z_k) * np.maximum(0, np.abs(z_k) - eta * lambda_val)
        
        # 収束判定
        if np.linalg.norm(x) > 1e-9 and np.linalg.norm(x_new - x) / np.linalg.norm(x) < tol:
            break
            
        x_prev = x
        x = x_new
        t_prev = t
        t = (1 + np.sqrt(1 + 4 * t**2)) / 2
        
    return x

# 半径τのL1ノルム球に射影する
def project_l1_ball(v, tau):
    if np.linalg.norm(v, 1) <= tau:
        return v
    
    u = np.abs(v)
    u_sorted = np.sort(u)[::-1]
    
    sv = np.cumsum(u_sorted)
    rho = np.where(u_sorted > (sv - tau) / (np.arange(len(u_sorted)) + 1))[0][-1]
    theta = (sv[rho] - tau) / (rho + 1)
    
    return np.sign(v) * np.maximum(0, u - theta)

# 制約付きLasso(LS_τ)をFISTAで解く
# LS_τ : min ||y - Ax||_2^2 subject to ||x||_1 <= tau
def solve_ls_fista(y, A, tau_val, max_iter=1000, tol=1e-6):
    m, n = A.shape
    x = np.zeros(n)
    x_prev = np.zeros(n)
    t = 1.0
    t_prev = 1.0
    
    # 目的関数 f(x) = ||y - Ax||^2 のリプシッツ定数
    L = 2 * np.linalg.norm(A.T @ A, 2)
    eta = 1.0 / L

    for i in range(max_iter):
        y_k = x + ((t_prev - 1) / t) * (x - x_prev)
        grad_f = 2 * A.T @ (A @ y_k - y)
        z_k = y_k - eta * grad_f
        x_new = project_l1_ball(z_k, tau_val) 
        
        if np.linalg.norm(x) > 1e-9 and np.linalg.norm(x_new - x) / np.linalg.norm(x) < tol:
            break
            
        x_prev = x
        x = x_new
        t_prev = t
        t = (1 + np.sqrt(1 + 4 * t**2)) / 2
        
    return x

# 基底追跡(BP_σ)をADMMで解く
# BP_σ : min ||x||_1  s.t. ||y - Ax||_2 <= sigma
def solve_bp_admm(y, A, sigma_val, rho=1.0, max_iter=1000, tol=1e-6):
    m, n = A.shape
    x = np.zeros(n)
    z1 = np.zeros(n)
    z2 = np.zeros(m)
    w1 = np.zeros(n)
    w2 = np.zeros(m)
    
    # xの更新式で使う逆行列を事前に計算
    inv_matrix = np.linalg.inv(np.eye(n) + A.T @ A)

    def soft_thresh(vec, t):
        return np.sign(vec) * np.maximum(np.abs(vec) - t, 0)
    
    for k in range(max_iter):
        x_prev = x.copy()

        # xの更新
        x = inv_matrix @ (z1 - w1 + A.T @ (z2 - w2))
        
        # z1の更新 (Soft-thresholding)
        z1 = soft_thresh(x + w1, 1.0 / rho)
        
        # z2の更新 (l2-ball projection)
        v = A @ x + w2
        dist_v_y = np.linalg.norm(v - y)
        if dist_v_y <= sigma_val:
            z2 = v
        else:
            z2 = y + sigma_val * (v - y) / dist_v_y
            
        # 双対変数 w1, w2 の更新
        w1 += x - z1
        w2 += A @ x - z2
        
        if np.linalg.norm(x_prev) > 1e-9 and np.linalg.norm(x - x_prev) / np.linalg.norm(x_prev) < tol:
            break
            
    return x

# メイン処理
if __name__ == '__main__':
    # 基本設定
    SAVE_DIR = '/home/user/program/Parameter-sensitivity/Lasso/result_lasso'
    FIG_SAVE_DIR = os.path.join(SAVE_DIR, 'figure')
    os.makedirs(FIG_SAVE_DIR, exist_ok=True)
    SUM_SAVE_DIR = os.path.join(SAVE_DIR, 'table')
    os.makedirs(SUM_SAVE_DIR, exist_ok=True)
    
    # シミュレーションのパラメータ
    s = 5      # 非ゼロ要素数
    M = 250     # 観測信号の数
    N = 1000    # 元信号の数
    noise_level = 0.03 # ノイズレベル
    
    # データの生成
    np.random.seed(0)  # 再現性のためのシード固定
    # 元信号 x0 を生成
    x0 = np.zeros(N)
    non_zero_indices = np.random.choice(N, s, replace=False)
    x0[non_zero_indices] = 1.0
    
    # 観測行列 A とノイズ v を生成
    A = np.random.randn(M, N)
    v = np.random.randn(M)
    
    # 観測データ y を生成
    y = A @ x0 + noise_level * v
    
################### ノイズレベル 0.03 ################
    # 各手法のパラメータ範囲を指定(重要)
    # No constraint : λ
    start_lambda = 0.09
    end_lambda = 20
    LAM_array = np.logspace(np.log10(start_lambda), np.log10(end_lambda), num=100)

    # TV constraint : τ
    start_tau = 0.4
    end_tau = 55
    TAU_array = np.logspace(np.log10(start_tau), np.log10(end_tau), num=100)

    # Data fidelity constraint : σ
    start_sigma = 0.02
    end_sigma = 7
    SIGMA_array = np.logspace(np.log10(start_sigma), np.log10(end_sigma), num=100)
    
    # 計算実行
    results = {'qp': {}, 'ls': {}, 'bp': {}}
    solvers = {'qp': solve_qp_fista, 'ls': solve_ls_fista, 'bp': solve_bp_admm}
    params = {'qp': LAM_array, 'ls': TAU_array, 'bp': SIGMA_array}
    
    for method, solver_func in solvers.items():
        print(f"Processing {method.upper()}...")
        for p_val in params[method]:
            if method == 'bp':
                restored = solver_func(y, A, p_val, rho=1.0)
            else:
                restored = solver_func(y, A, p_val)
            
            results[method][p_val] = {
                'mse': mean_squared_error(x0, restored),
                'psnr': peak_signal_noise_ratio(x0, restored, data_range=1.0),
                'ssim': structural_similarity(x0, restored, data_range=1.0)
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
            'ssim': [results[method][p]['ssim'] for p in params[method]]
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
        
        # パラメータ軸を正規化
        normalized_axes_psnr[method] = params[method] / optimal_params[method]['psnr']
        normalized_axes_ssim[method] = params[method] / optimal_params[method]['ssim']
        print(f"Optimal for {method.upper()}: PSNR-based={optimal_params[method]['psnr']:.4f}, SSIM-based={optimal_params[method]['ssim']:.4f}, MSE-based={optimal_params[method]['mse']:.4f}")

    # 結果のテキストファイル出力
    summary_text = "Lasso Denoising Results Summary\n" + "="*50 + "\n\n"
    summary_text += f"Signal: N={N}, M={M}, s={s}, noise_level={noise_level}\n\n"
    for method in ['qp', 'ls', 'bp']:
        summary_text += f"--- {method.upper()} Optimal Parameters ---\n"
        summary_text += f"Based on Min MSE   : {optimal_params[method]['mse']:.6f}\n"
        summary_text += f"Based on Max PSNR  : {optimal_params[method]['psnr']:.6f}\n"
        summary_text += f"Based on Max SSIM  : {optimal_params[method]['ssim']:.6f}\n\n"
    summary_file_path = os.path.join(SUM_SAVE_DIR, f'optimal_parameters_summary_noise{noise_level}_s{s}_M{M}_N{N}.txt')
    with open(summary_file_path, 'w') as f: f.write(summary_text)
    print(f"\nOptimal parameters summary saved to {summary_file_path}")

    # 感度グラフ描画
    metrics_to_plot = {
        'MSE': {'data_key': 'mse', 'plot_func': plt.loglog, 'ylabel': 'MSE'},
        'PSNR': {'data_key': 'psnr', 'plot_func': plt.semilogx, 'ylabel': 'PSNR (dB)'},
        'SSIM': {'data_key': 'ssim', 'plot_func': plt.semilogx, 'ylabel': 'SSIM'}
    }
    
    colors = {'ls': 'green', 'bp': 'orange', 'qp': 'blue'}
    labels = {'ls': 'L1-norm constraint ($\\tau$)', 'bp': 'Data fidelity constraint ($\sigma$)', 'qp': 'No constraint ($\lambda$)'}

    for metric_name, plot_info in metrics_to_plot.items():
        plt.figure(figsize=(10, 7))
        
        for method in ['ls', 'bp', 'qp']:
            if metric_name == 'SSIM':
                x_axis_data = normalized_axes_ssim[method]
            else:
                x_axis_data = normalized_axes_psnr[method]
            
            plot_info['plot_func'](
                x_axis_data,
                final_metrics[method][plot_info['data_key']],
                color=colors[method],
                label=labels[method],
                linewidth=3
            )

        plt.xlabel('Normalized Parameter', fontsize=18)
        plt.ylabel(plot_info['ylabel'], fontsize=18)
        plt.tick_params(labelsize=16)
        plt.legend(fontsize=16)
        # plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.xlim(1e-1, 1e1)      
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_SAVE_DIR, f'parameter_sensitivity_lasso_{metric_name}_noise{noise_level}_s{s}_M{M}_N{N}.pdf'))
        plt.show()

    # 実行完了通知
    end_time = time.time()
    hours, rem = divmod(end_time - start_time, 3600)
    minutes, seconds = divmod(rem, 60)
    exec_time_str = f"実行時間: {int(hours)}時間 {int(minutes)}分 {int(seconds)}秒"
    data = {"content": f"\nLassoプログラム全処理完了! (SSIM評価含む)\n{exec_time_str}"}
    requests.post(DISCORD_WEBHOOK_URL, data=data)