# plot_graph.py
# ノイズを含む画像復元(y=x+v) - グラフ描画編

import os
import json
import numpy as np
import matplotlib.pyplot as plt

# グラフ全体のフォント設定
plt.rcParams['font.family'] = 'serif'
plt.rcParams['mathtext.fontset'] = 'cm' 

def plot_from_json(json_file_path, save_dir):
    # JSONデータの読み込み
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    meta = data['meta']
    results = data['results']
    
    noise_level = meta['noise_level']
    image_filename = meta['image_filename']
    
    print(f"Loaded data for {image_filename} (Noise: {noise_level})")
    
    # データ構築と最適値の再計算（正規化軸のため）
    final_metrics = {'qp': {}, 'ls': {}, 'bp': {}}
    optimal_params = {'qp': {}, 'ls': {}, 'bp': {}}
    normalized_axes_psnr = {}
    normalized_axes_ssim = {}
    
    methods = ['qp', 'ls', 'bp']
    
    for method in methods:
        m_data = results[method]
        
        # リストをnumpy配列に戻す
        params = np.array(m_data['params'])
        metrics = {
            'mse': np.array(m_data['mse']),
            'psnr': np.array(m_data['psnr']),
            'ssim': np.array(m_data['ssim']),
            'tv': np.array(m_data['tv'])
        }
        final_metrics[method] = metrics
        
        # 最適パラメータのインデックス探索
        idx_mse = np.argmin(metrics['mse'])
        idx_psnr = np.argmax(metrics['psnr'])
        idx_ssim = np.argmax(metrics['ssim'])
        
        opt_p = {
            'mse': params[idx_mse],
            'psnr': params[idx_psnr],
            'ssim': params[idx_ssim]
        }
        optimal_params[method] = opt_p
        
        # 軸の正規化データの作成
        normalized_axes_psnr[method] = params / opt_p['psnr']
        normalized_axes_ssim[method] = params / opt_p['ssim']
        
        print(f"{method.upper()} Optimals -> PSNR: {opt_p['psnr']:.4f}, SSIM: {opt_p['ssim']:.4f}, MSE: {opt_p['mse']:.4f}")

    # 結果サマリーテキストの保存
    summary_text = f"ADMM TV Denoising Summary for {image_filename}\n"
    summary_text += "="*70 + "\n\n"
    summary_text += f"Original TV: {meta['tv_orig']:.2f}\n"
    summary_text += f"Noisy TV:    {meta['tv_noisy']:.2f}\n\n"
    
    for method in methods:
        summary_text += f"--- {method.upper()} Optimal Parameters ---\n"
        summary_text += f"Based on Min MSE   : {optimal_params[method]['mse']:.6f}\n"
        summary_text += f"Based on Max PSNR  : {optimal_params[method]['psnr']:.6f}\n"
        summary_text += f"Based on Max SSIM  : {optimal_params[method]['ssim']:.6f}\n\n"
    
    sum_save_path = os.path.join(save_dir, 'table')
    os.makedirs(sum_save_path, exist_ok=True)
    with open(os.path.join(sum_save_path, f'optimal_parameters_summary_noise_{noise_level}_{image_filename}.txt'), 'w') as f:
        f.write(summary_text)

    # グラフ描画設定
    fig_save_dir = os.path.join(save_dir, 'figure')
    os.makedirs(fig_save_dir, exist_ok=True)
    
    metrics_to_plot = {
        'MSE': {'data_key': 'mse', 'plot_func': plt.loglog, 'ylabel': 'Mean Squared Error'},
        'PSNR': {'data_key': 'psnr', 'plot_func': plt.semilogx, 'ylabel': 'PSNR (dB)'},
        'SSIM': {'data_key': 'ssim', 'plot_func': plt.semilogx, 'ylabel': 'SSIM'}
    }
    
    colors = {'qp': 'blue', 'ls': 'green', 'bp': 'orange'}
    labels = {'qp': r'No constraint ($\lambda$)', 'ls': r'TV constraint ($\tau$)', 'bp': r'Data fidelity constraint ($\sigma$)'}
    linestyles = {'qp': '-', 'ls': '--', 'bp': ':'}

    for metric_name, plot_info in metrics_to_plot.items():
        plt.figure(figsize=(10, 7))
        
        for method in methods:
            # SSIMのグラフだけSSIM基準の正規化軸を使用、他はPSNR基準
            if metric_name == 'SSIM':
                x_axis_data = normalized_axes_ssim[method]
            else:
                x_axis_data = normalized_axes_psnr[method]
            
            y_axis_data = final_metrics[method][plot_info['data_key']]
            
            plot_info['plot_func'](
                x_axis_data,
                y_axis_data,
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
        
        # 保存
        filename_base = f'parameter_sensitivity_tv_admm_{metric_name}_noise_{noise_level}_{image_filename}'
        plt.savefig(os.path.join(fig_save_dir, f'{filename_base}.pdf'))
        plt.savefig(os.path.join(fig_save_dir, f'{filename_base}.png'))
        print(f"Saved plot: {filename_base}")
        # plt.show() # 自動処理の場合はコメントアウト推奨

# --- メイン処理 ---
if __name__ == '__main__':
    # JSONファイルのパスを指定
    # ※ main_process.py で生成されたJSONのパスに合わせてください
    JSON_PATH = '/home/user/program/Parameter-sensitivity/Total_Variation/result_tv_admm_denoising/data/results_noise_0.05_cameraman.png.json'
    SAVE_ROOT = '/home/user/program/Parameter-sensitivity/Total_Variation/result_tv_admm_denoising'
    
    if os.path.exists(JSON_PATH):
        plot_from_json(JSON_PATH, SAVE_ROOT)
    else:
        print(f"Error: JSON file not found at {JSON_PATH}")