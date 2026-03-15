# ノイズとぼけを含む画像復元(y=Ax+v)
# Lassoを用いた最適化問題を解く (SSIM評価あり)
# num_trials回の平均の復元結果

import os
import numpy as np
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity
import matplotlib.pyplot as plt
import time
import requests

# 保存と通知の設定
DISCORD_WEBHOOK_URL = 'https://discord.com/api/webhooks/1313038750023942204/I_bKSXATJu0IsHH6Fi9tgMXn67KkEMba94WH5XE3SE-bwoImQIEKzTss_zHIqnRFvTew'
# タイマー開始
start_time = time.time()


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
    noise_level = 0.05 # ノイズレベル

    # シミュレーションの試行回数
    num_trials = 100
        
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

################### ノイズレベル 0.05 ################
    # 各手法のパラメータ範囲を指定(重要)
    # No constraint (QP) : λ
    start_lambda = 0.1
    end_lambda = 35
    LAM_array = np.logspace(np.log10(start_lambda), np.log10(end_lambda), num=100)

    # TV constraint (LS) : τ
    start_tau = 0.4
    end_tau = 55
    TAU_array = np.logspace(np.log10(start_tau), np.log10(end_tau), num=100)

    # Data fidelity constraint : σ
    start_sigma = 0.06
    end_sigma = 15
    SIGMA_array = np.logspace(np.log10(start_sigma), np.log10(end_sigma), num=100)

    # 平均化のための共通正規化軸
    common_norm_axis = np.logspace(-1, 1, num=200)
    
    # 全試行の結果を保存する辞書
    all_curves = {
        method: {
            metric: np.zeros((num_trials, len(common_norm_axis)))
            for metric in ['mse', 'psnr', 'ssim']
        } for method in ['qp', 'ls', 'bp']
    }
    all_optimal_params = {
        method: {
            metric: [] for metric in ['mse', 'psnr', 'ssim']
        } for method in ['qp', 'ls', 'bp']
    }
    
    # 計算実行
    solvers = {'qp': solve_qp_fista, 'ls': solve_ls_fista, 'bp': solve_bp_admm}
    params = {'qp': LAM_array, 'ls': TAU_array, 'bp': SIGMA_array}
    
    # 複数回試行するための外側ループ
    for i in range(num_trials):
        print(f"--- Running Trial {i+1}/{num_trials} ---")
        
        # この試行の結果を一時的に保存
        results = {'qp': {}, 'ls': {}, 'bp': {}}

        # 計算実行
        for method, solver_func in solvers.items():
            print(f"  Processing {method.upper()}...")
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
        
        # この試行での最適パラメータを決定
        for method in ['qp', 'ls', 'bp']:
            final_metrics = {
                'mse': [results[method][p]['mse'] for p in params[method]],
                'psnr': [results[method][p]['psnr'] for p in params[method]],
                'ssim': [results[method][p]['ssim'] for p in params[method]]
            }
            
            idx_psnr = np.argmax(final_metrics['psnr'])
            idx_ssim = np.argmax(final_metrics['ssim'])
            idx_mse = np.argmin(final_metrics['mse'])
            
            optimal_psnr_param = params[method][idx_psnr]
            optimal_ssim_param = params[method][idx_ssim]
            optimal_mse_param = params[method][idx_mse]

            all_optimal_params[method]['psnr'].append(optimal_psnr_param)
            all_optimal_params[method]['ssim'].append(optimal_ssim_param)
            all_optimal_params[method]['mse'].append(optimal_mse_param)

            # この試行の正規化軸を作成し、共通軸上に補間して保存
            normalized_axis_psnr = params[method] / optimal_psnr_param
            normalized_axis_ssim = params[method] / optimal_ssim_param
            
            all_curves[method]['mse'][i, :] = np.interp(common_norm_axis, normalized_axis_psnr, final_metrics['mse'])
            all_curves[method]['psnr'][i, :] = np.interp(common_norm_axis, normalized_axis_psnr, final_metrics['psnr'])
            all_curves[method]['ssim'][i, :] = np.interp(common_norm_axis, normalized_axis_ssim, final_metrics['ssim'])

    # --- 全試行終了後、平均を計算 ---
    print("\n--- Averaging Results Across All Trials ---")
    mean_curves = {
        method: {
            metric: np.mean(all_curves[method][metric], axis=0)
            for metric in ['mse', 'psnr', 'ssim']
        } for method in ['qp', 'ls', 'bp']
    }

    avg_optimal_params = {
        method: {
            metric: np.mean(all_optimal_params[method][metric])
            for metric in ['mse', 'psnr', 'ssim']
        } for method in ['qp', 'ls', 'bp']
    }

    # 結果のテキストファイル出力 (平均値を使用)
    summary_text = f"Lasso Denoising Results Summary (Averaged over {num_trials} trials)\n" + "="*70 + "\n\n"
    summary_text += f"Signal: N={N}, M={M}, s={s}, noise_level={noise_level}\n"
    summary_text += f"Note: x0 and A were fixed across all trials.\n\n"
    for method in ['qp', 'ls', 'bp']:
        summary_text += f"--- {method.upper()} Average Optimal Parameters ---\n"
        summary_text += f"Based on Min MSE   : {avg_optimal_params[method]['mse']:.6f}\n"
        summary_text += f"Based on Max PSNR  : {avg_optimal_params[method]['psnr']:.6f}\n"
        summary_text += f"Based on Max SSIM  : {avg_optimal_params[method]['ssim']:.6f}\n\n"
    summary_file_path = os.path.join(SUM_SAVE_DIR, f'optimal_parameters_summary_noise{noise_level}_s{s}_M{M}_N{N}_avg_fixed.txt')
    with open(summary_file_path, 'w') as f: f.write(summary_text)
    print(f"\nAverage optimal parameters summary saved to {summary_file_path}")

    # 感度グラフ描画 (平均値を使用)
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
            plt.plot(
                common_norm_axis,
                mean_curves[method][plot_info['data_key']],
                color=colors[method],
                label=labels[method],
                linewidth=3
            )
        
        # プロット関数を動的に適用
        if plot_info['plot_func'] == plt.loglog:
            plt.xscale('log')
            plt.yscale('log')
        elif plot_info['plot_func'] == plt.semilogx:
            plt.xscale('log')

        plt.xlabel('Normalized Parameter', fontsize=18)
        plt.ylabel(plot_info['ylabel'], fontsize=18)
        plt.tick_params(labelsize=16)
        plt.legend(fontsize=16)
        # plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.xlim(1e-1, 1e1)
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_SAVE_DIR, f'parameter_sensitivity_lasso_{metric_name}_noise{noise_level}_s{s}_M{M}_N{N}_avg_fixed.pdf'))
        plt.show()

    # 実行完了通知
    end_time = time.time()
    hours, rem = divmod(end_time - start_time, 3600)
    minutes, seconds = divmod(rem, 60)
    exec_time_str = f"実行時間: {int(hours)}時間 {int(minutes)}分 {int(seconds)}秒"
    data = {"content": f"\nLasso全処理完了! ({num_trials}回の試行平均, x0とA固定)\n{exec_time_str}"}
    requests.post(DISCORD_WEBHOOK_URL, data=data)