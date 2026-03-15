# インペインティング（欠損）を含む画像復元
# 全変動正則化を用いた最適化問題をADMMで解く

import os
import numpy as np
import cv2
from scipy.fftpack import fft2, ifft2
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity
import matplotlib.pyplot as plt
from numpy.fft import fft2 as _fft2, ifft2 as _ifft2
import time
import requests

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
def project_l2_ball(v, y, mask, sigma):
    diff = mask * (v - y) # マスクされた領域でのみvとyの差を計算
    dist = np.linalg.norm(diff)
    
    # 制約を満たしていればvをそのまま返す
    if dist <= sigma:
        return v
    # 制約を超えている場合、マスク領域の差分ベクトルを補正して射影
    else:
        correction = diff * (1 - sigma / dist)
        return v - correction

# Total Variationを計算する関数
def calculate_tv(image):
    # np.gradientは(dy, dx)の順で勾配を返す
    dy, dx = np.gradient(image)
    tv = np.sum(np.sqrt(dx**2 + dy**2))
    return tv

# D^T D（ラプラシアン）作用 と D^T の適用
def apply_L(x, otf_dx, otf_dy):
    # L x = D^T D x
    Fx = _fft2(x)
    return np.real(_ifft2((np.conj(otf_dx)*otf_dx + np.conj(otf_dy)*otf_dy) * Fx))

def apply_Dt(bx, by, otf_dx, otf_dy):
    # D^T [bx, by]
    return np.real(_ifft2(np.conj(otf_dx) * _fft2(bx) + np.conj(otf_dy) * _fft2(by)))

# 共役勾配法
def cg_solve(Afun, b, x0=None, max_iter=50, tol=1e-6):
    x = np.zeros_like(b) if x0 is None else x0.copy()
    r = b - Afun(x)
    p = r.copy()
    rs_old = np.sum(r*r)
    for _ in range(max_iter):
        Ap = Afun(p)
        alpha = rs_old / (np.sum(p*Ap) + 1e-12)
        x += alpha * p
        r -= alpha * Ap
        rs_new = np.sum(r*r)
        if np.sqrt(rs_new) < tol:
            break
        p = r + (rs_new/rs_old) * p
        rs_old = rs_new
    return x

#  v_obs, y_obs は画像と同サイズの配列だが, ノルム・投影は mask==1 の画素に対してのみ評価

def project_l2_ball_masked(v_obs, y_obs, sigma, mask):
    diff = (v_obs - y_obs) * mask
    dist = np.linalg.norm(diff)
    if dist <= sigma:
        # 観測以外は0のままでよい（v_obsもそのまま返す）
        return v_obs * mask
    # 観測画素上でだけスケール
    return (y_obs + sigma * diff / (dist + 1e-12)) * mask

#### 各手法のADMM ####
# No constraint: min_x 1/2*||Ax - y||_2^2 + lam * TV(x)
def tv_admm_qp(y, mask, lam, rho, num_iterations):
    img_shape = y.shape
    x = y.copy()
    z_x, z_y, u_x, u_y = [np.zeros(img_shape) for _ in range(4)]
    otf_dx, otf_dy = get_difference_operators(img_shape)

    # x-ステップ: (A^T A + rho L) x = A^T y + rho D^T(z-u)
    def Afun(x_):
        return mask * x_ + rho * apply_L(x_, otf_dx, otf_dy)

    rhs_const = mask * y  # A^T y (Aは対角なので同じ)
    for _ in range(num_iterations):
        # 右辺の D^T(z-u)
        Dt_term = apply_Dt(z_x - u_x, z_y - u_y, otf_dx, otf_dy)
        b = rhs_const + rho * Dt_term

        # CG で線形方程式を解く
        x = cg_solve(Afun, b, x0=x, max_iter=50, tol=1e-6)

        # z-ステップ（そのまま）
        dx_x = np.real(_ifft2(otf_dx * _fft2(x)))
        dy_x = np.real(_ifft2(otf_dy * _fft2(x)))
        v_x, v_y = dx_x + u_x, dy_x + u_y
        z_x, z_y = shrink(v_x, v_y, lam / rho)

        # u-ステップ（そのまま）
        u_x += (dx_x - z_x)
        u_y += (dy_x - z_y)

    x = np.clip(x, 0, 1)

    return x


# TV constraint: min_x ||Ax - y||_2^2  subject to  TV(x) <= tau
# 旧: def tv_admm_ls(noisy_img, tau, rho, num_iterations):
def tv_admm_ls(y, mask, tau, rho, num_iterations):
    img_shape = y.shape
    x = y.copy()
    z_x, z_y, u_x, u_y = [np.zeros(img_shape) for _ in range(4)]
    otf_dx, otf_dy = get_difference_operators(img_shape)

    # x-ステップ: (2 A^T A + rho L) x = 2 A^T y + rho D^T(z-u)
    def Afun(x_):
        return 2.0 * (mask * x_) + rho * apply_L(x_, otf_dx, otf_dy)

    rhs_const = 2.0 * (mask * y)
    for _ in range(num_iterations):
        Dt_term = apply_Dt(z_x - u_x, z_y - u_y, otf_dx, otf_dy)
        b = rhs_const + rho * Dt_term
        x = cg_solve(Afun, b, x0=x, max_iter=50, tol=1e-6)

        dx_x = np.real(_ifft2(otf_dx * _fft2(x)))
        dy_x = np.real(_ifft2(otf_dy * _fft2(x)))
        v_x, v_y = dx_x + u_x, dy_x + u_y
        z_x, z_y = project_l12_ball(v_x, v_y, tau)

        u_x += (dx_x - z_x); u_y += (dy_x - z_y)

    x = np.clip(x, 0, 1)
    
    return x


# Data fidelity constraint: min_x TV(x)  subject to  ||Ax - y||_2 <= sigma
# 旧: def tv_admm_bp(noisy_img, sigma, rho, num_iterations):
def tv_admm_bp(y, mask, sigma, rho, num_iterations):
    img_shape = y.shape
    x = y.copy()

    # 変数: z1=(TV用), z2=(Ax用)
    z1_x, z1_y, z2 = [np.zeros(img_shape) for _ in range(3)]
    u1_x, u1_y, u2 = [np.zeros(img_shape) for _ in range(3)]

    otf_dx, otf_dy = get_difference_operators(img_shape)

    # x-ステップ: (A^T A + L) x = A^T(z2 - u2) + D^T(z1 - u1)
    def Afun(x_):
        return (mask * x_) + apply_L(x_, otf_dx, otf_dy)

    for _ in range(num_iterations):
        Dt_term = apply_Dt(z1_x - u1_x, z1_y - u1_y, otf_dx, otf_dy)
        b = (mask * (z2 - u2)) + Dt_term
        x = cg_solve(Afun, b, x0=x, max_iter=50, tol=1e-6)

        # z1（TVの縮小）: しきい値は 1/rho（そのまま）
        dx_x = np.real(_ifft2(otf_dx * _fft2(x)))
        dy_x = np.real(_ifft2(otf_dy * _fft2(x)))
        v1_x, v1_y = dx_x + u1_x, dy_x + u1_y
        z1_x, z1_y = shrink(v1_x, v1_y, 1.0 / rho)

        # z2（観測空間での l2 球への射影）
        v2 = (mask * x) + u2  # s = A x の分割変数
        z2 = project_l2_ball_masked(v2, y, sigma, mask)

        # 乗数更新
        u1_x += (dx_x - z1_x)
        u1_y += (dy_x - z1_y)
        u2   += (mask * x - z2)

    x = np.clip(x, 0, 1)

    return x


# メイン処理
if __name__ == '__main__':
        # 基本設定
        IMG_PATH = '/home/user/program/Parameter-sensitivity/Total_Variation/image/cameraman.png'
        SAVE_DIR = '/home/user/program/Parameter-sensitivity/Total_Variation/result_tv_admm_inpainting'
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
        np.random.seed(0)

        # マスクの種類と欠損率を設定
        mask_type = 'random' # 'random' または 'center' を選択
        p_miss = 0.5 # 欠損率

        # ノイズを生成し、ノイズ画像をクリッピング
        noise = np.random.randn(*orig_img.shape) * noise_level
        noisy_img = np.clip(orig_img + noise, 0, 1)

        # mask_typeに応じてマスクとファイル名を生成
        if mask_type == 'random':
            mask = (np.random.rand(*orig_img.shape) > p_miss).astype(np.float64)
            filename_suffix = f'random_mask_miss{p_miss}_noise{noise_level}'
            
        elif mask_type == 'center':
            h, w = orig_img.shape
            # p_miss に基づいて欠損させる総ピクセル数を計算
            total_pixels = h * w
            missing_pixels_target = int(total_pixels * p_miss)
            
            # ターゲットのピクセル数に最も近い正方形の一辺の長さを計算
            side_length = int(np.sqrt(missing_pixels_target))
            half_side = side_length // 2
            
            # マスクを生成
            mask = np.ones_like(orig_img)
            center_y, center_x = h // 2, w // 2
            start_y, end_y = center_y - half_side, center_y + (side_length - half_side)
            start_x, end_x = center_x - half_side, center_x + (side_length - half_side)
            mask[start_y:end_y, start_x:end_x] = 0
            
            # 実際の欠損率を計算
            actual_miss_rate = 1 - np.mean(mask)
            print(f"中央マスクのターゲット欠損率: {p_miss:.4f}")
            print(f"中央マスクの実際の欠損率: {actual_miss_rate:.4f} ({side_length}x{side_length}の領域)")

            filename_suffix = f'center_mask_miss{p_miss:.1f}_noise{noise_level}'
        else:
            raise ValueError("無効なmask_typeです。'random' または 'center' を選択してください。")


        # 欠損画像の作成と保存
        inpainted_img = noisy_img * mask
        cv2.imwrite(os.path.join(IMG_SAVE_DIR, f'inpainted_{filename_suffix}.png'), (inpainted_img * 255).astype(np.uint8))

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
        start_sigma = 0.5
        end_sigma = 150
        SIGMA_array = np.logspace(np.log10(start_sigma), np.log10(end_sigma), num=50)

        # 計算実行
        results = {'qp': {}, 'ls': {}, 'bp': {}}
        solvers = {'qp': tv_admm_qp, 'ls': tv_admm_ls, 'bp': tv_admm_bp}
        params = {'qp': LAM_array, 'ls': TAU_array, 'bp': SIGMA_array}

        for method, solver_func in solvers.items():
            print(f"Processing {method.upper()}...")
            for p_val in params[method]:
                restored = solvers[method](inpainted_img, mask, p_val, RHO, EP)
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
                cv2.imwrite(os.path.join(IMG_SAVE_DIR, f"restored_{method}_noise{noise_level}_loss_rate_{p_miss}_np{scale:.0e}_{image_filename}"), (restored_img * 255).astype(np.uint8))

        # 結果のテキストファイル出力
        summary_text = "ADMM TV Inpainting Results Summary\n" + "="*50 + "\n\n" # <<< 修正: Denoising -> Inpainting
        for method in ['qp', 'ls', 'bp']:
            summary_text += f"--- {method.upper()} Optimal Parameters ---\n"
            summary_text += f"Based on Min MSE  : {optimal_params[method]['mse']:.6f}\n"
            summary_text += f"Based on Max PSNR : {optimal_params[method]['psnr']:.6f}\n"
            summary_text += f"Based on Max SSIM : {optimal_params[method]['ssim']:.6f}\n\n"
        summary_file_path = os.path.join(SUM_SAVE_DIR, f'optimal_parameters_summary_{filename_suffix}.txt')
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
            plt.savefig(os.path.join(FIG_SAVE_DIR, f'parameter_sensitivity_tv_admm_{metric_name}_noise_{noise_level}_loss_rate_{p_miss}_{filename_suffix}_inpainting.pdf'))
            plt.savefig(os.path.join(FIG_SAVE_DIR, f'parameter_sensitivity_tv_admm_{metric_name}_noise_{noise_level}_loss_rate_{p_miss}_{filename_suffix}_inpainting.png'))
            plt.show()

        # except Exception as e:
        #     # ### エラー発生時の処理 ###
        #     end_time = time.time()
        #     hours, rem = divmod(end_time - start_time, 3600)
        #     minutes, seconds = divmod(rem, 60)
        #     exec_time_str = f"実行時間: {int(hours)}時間 {int(minutes)}分 {int(seconds)}秒"
            
        #     # エラー情報とトレースバックを取得
        #     tb_str = traceback.format_exc()
            
        #     # Discordに送信するエラーメッセージを作成
        #     error_message = (
        #         f"**プログラムの実行中にエラーが発生しました。**\n"
        #         f"{exec_time_str}\n\n"
        #         f"**エラータイプ:** `{type(e).__name__}`\n"
        #         f"**エラーメッセージ:** `{e}`\n\n"
        #         f"**トレースバック:**\n"
        #         f"```\n{tb_str}\n```"
        #     )
            
        #     # Discordの文字数制限(2000文字)を超えないように調整
        #     if len(error_message) > 1990:
        #         error_message = error_message[:1990] + "...\n```"
                
        #     data = {"content": error_message}
        #     requests.post(DISCORD_WEBHOOK_URL, data=data)
            
        # else:
        #     # ### 正常終了時の処理 ###
        #     end_time = time.time()
        #     hours, rem = divmod(end_time - start_time, 3600)
        #     minutes, seconds = divmod(rem, 60)
        #     exec_time_str = f"実行時間: {int(hours)}時間 {int(minutes)}分 {int(seconds)}秒"
            
        #     # 正常完了のメッセージを送信
        #     data = {"content": f"✅ **tv全処理完了!**\n{exec_time_str}"}
        #     requests.post(DISCORD_WEBHOOK_URL, data=data)
  
# 実行完了通知
end_time = time.time()
hours, rem = divmod(end_time - start_time, 3600)
minutes, seconds = divmod(rem, 60)
exec_time_str = f"実行時間: {int(hours)}時間 {int(minutes)}分 {int(seconds)}秒"
data = {"content": f"\nインペインティング全処理完了!\n{exec_time_str}"}
requests.post(DISCORD_WEBHOOK_URL, data=data) 