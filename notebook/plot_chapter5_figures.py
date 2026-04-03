# -*- coding: utf-8 -*-
"""
第五章 图11 & 图12 绘制脚本
字体方案：中文宋体（Songti SC）作为全局默认，
         英文/数字元素通过 fontproperties 单独指定 Times New Roman
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.font_manager as fm
import numpy as np
import os

# ============================================================
# 字体配置
# 将 Songti SC 设为全局默认，保证中文不出方块
# 英文部分通过 fontproperties 指定 Times New Roman
# ============================================================
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Songti SC', 'LiSong Pro', 'STFangsong', 'DejaVu Sans'],
    'axes.unicode_minus': False,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'font.size': 13,
    'axes.labelsize': 14,
    'axes.titlesize': 15,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
})

# FontProperties 对象
FP_CN       = fm.FontProperties(family='Songti SC', size=12)
FP_CN_BOLD  = fm.FontProperties(family='Songti SC', size=12, weight='bold')
FP_CN_SM    = fm.FontProperties(family='Songti SC', size=11)
FP_CN_TITLE = fm.FontProperties(family='Songti SC', size=15)
FP_CN_GRP   = fm.FontProperties(family='Songti SC', size=14, weight='bold')
FP_EN       = fm.FontProperties(family='Times New Roman', size=12)
FP_EN_BOLD  = fm.FontProperties(family='Times New Roman', size=12, weight='bold')

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'document', 'figures')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# 数据
# ============================================================
ablation_labels = ['基准模型\n(逻辑回归)', '实验A\n(ResNet编码器)', '实验B\n(+对比学习预训练)', '实验C\n(+盈利排序微调)']
ablation_means  = [505747.90, 555645.60, 566869.56, 591740.14]
ablation_stds   = [6879.03, 41137.57, 25523.73, 29119.38]

all_labels = [
    '基准模型\n(逻辑回归)',
    '实验A\n(ResNet)',
    '实验B\n(+对比预训练)',
    '实验C\n(+盈利排序)',
    'LightGBM',
    'XGBoost',
    '编码器\n+LightGBM',
    '编码器\n+XGBoost',
]
all_means = [505747.90, 555645.60, 566869.56, 591740.14,
             422179.87, 462813.28, 518785.81, 486340.49]
all_stds  = [6879.03, 41137.57, 25523.73, 29119.38,
             33571.27, 45193.75, 27997.24, 34743.38]

COLOR_BASE  = '#9E9E9E'
COLOR_A     = '#90CAF9'
COLOR_B     = '#42A5F5'
COLOR_C     = '#C62828'
COLOR_TREE1 = '#66BB6A'
COLOR_TREE2 = '#43A047'
COLOR_HYB1  = '#FFA726'
COLOR_HYB2  = '#FB8C00'

ablation_colors = [COLOR_BASE, COLOR_A, COLOR_B, COLOR_C]
all_colors = [COLOR_BASE, COLOR_A, COLOR_B, COLOR_C,
              COLOR_TREE1, COLOR_TREE2, COLOR_HYB1, COLOR_HYB2]


# ============================================================
# 图11：消融实验递进提升
# ============================================================
def plot_fig11():
    fig, ax = plt.subplots(figsize=(10, 6.5))
    x = np.arange(len(ablation_labels))
    bar_width = 0.55

    bars = ax.bar(x, ablation_means, bar_width,
                  color=ablation_colors, edgecolor='white', linewidth=0.8,
                  yerr=ablation_stds, capsize=5,
                  error_kw=dict(elinewidth=1.4, capthick=1.4, color='#444444'))

    # 阶梯虚线 + 增量标注
    for i in range(1, len(ablation_means)):
        inc  = ablation_means[i] - ablation_means[i - 1]
        pct  = inc / ablation_means[i - 1] * 100
        y_p  = ablation_means[i - 1]

        ax.plot([x[i-1] + bar_width/2, x[i] - bar_width/2], [y_p, y_p],
                color='#888888', linestyle='--', linewidth=1.1)
        ax.plot([x[i] - bar_width/2, x[i] - bar_width/2], [y_p, ablation_means[i]],
                color='#888888', linestyle='--', linewidth=1.1)

        mid_x = (x[i-1] + x[i]) / 2
        mid_y = max(ablation_means[i], y_p) + max(ablation_stds[i], ablation_stds[i-1]) + 12000
        ax.annotate(f'+{inc/1e4:.2f}万  (+{pct:.1f}%)',
                    xy=(mid_x, mid_y), ha='center', va='bottom',
                    fontproperties=FP_CN_BOLD, color='#C62828',
                    bbox=dict(boxstyle='round,pad=0.35', facecolor='#FFF3F3',
                              edgecolor='#DDAAAA', alpha=0.95))

    # 柱内数值
    for i in range(len(ablation_means)):
        ax.text(x[i], (ablation_means[i] + 440000) / 2,
                f'{ablation_means[i]/1e4:.2f}万',
                ha='center', va='center',
                fontproperties=FP_CN_BOLD, color='white')

    # x 轴标签
    ax.set_xticks(x)
    ax.set_xticklabels(ablation_labels, fontproperties=FP_CN, linespacing=1.4)

    # y 轴标签（含中文，用宋体）
    ax.set_ylabel('Top-30% 累计总收益（元）', fontproperties=FP_CN_TITLE)
    # ax.set_title('图 11　消融实验各阶段 Top-30% 累计总收益递进提升',
    #              fontproperties=FP_CN_TITLE, pad=16)

    ax.set_ylim(440000, 730000)
    # y 轴刻度：数字用 Times New Roman，"万"不可避免走全局字体（宋体），无问题
    ax.yaxis.set_major_formatter(
        mpl.ticker.FuncFormatter(lambda v, _: f'{v/1e4:.0f}万'))

    ax.yaxis.grid(True, linestyle=':', alpha=0.5)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.subplots_adjust(bottom=0.15, top=0.89)
    path = os.path.join(OUTPUT_DIR, '图11_消融实验递进提升.png')
    fig.savefig(path, bbox_inches='tight')
    print(f'✓ 图11 已保存: {path}')
    plt.close(fig)


# ============================================================
# 图12：全部模型横向对比
# ============================================================
def plot_fig12():
    fig, ax = plt.subplots(figsize=(13, 7.5))
    positions = [0, 1, 2, 3, 4.8, 5.8, 7.0, 8.0]
    bar_width = 0.72
    y_bottom  = 360000

    bars = ax.bar(positions, all_means, bar_width,
                  color=all_colors, edgecolor='white', linewidth=0.8,
                  yerr=all_stds, capsize=5,
                  error_kw=dict(elinewidth=1.4, capthick=1.4, color='#444444'))

    # 基准线
    baseline_val = all_means[0]
    ax.axhline(y=baseline_val, color='#777777', linestyle='--', linewidth=1.3, alpha=0.8)
    ax.text(positions[-1] + 0.75, baseline_val + 2500,
            f'基准线  {baseline_val/1e4:.2f}万',
            fontproperties=FP_CN_SM, color='#555555', va='bottom')

    # 柱内/柱上数值
    for i in range(len(all_means)):
        val = all_means[i]
        if val > 480000:
            ax.text(positions[i], (val + y_bottom) / 2,
                    f'{val/1e4:.2f}万',
                    ha='center', va='center',
                    fontproperties=FP_CN_BOLD, color='white')
        else:
            ax.text(positions[i], val + all_stds[i] + 7000,
                    f'{val/1e4:.2f}万',
                    ha='center', va='bottom',
                    fontproperties=FP_CN_BOLD, color='#333333')

    # 分组背景色块
    for (xl, xr), bg in zip([(-0.5, 3.5), (4.3, 8.5)], ['#EEF4FF', '#EEFFF0']):
        ax.axvspan(xl, xr, color=bg, alpha=0.6, zorder=0)

    # 分组标签（放在背景色块内顶部，带边框）
    for cx, gn, gc in zip(
        [1.5, 6.4],
        ['消融实验组', '横向对比实验组'],
        ['#1565C0', '#2E7D32']
    ):
        ax.text(cx, 690000, gn,
                ha='center', va='center',
                fontproperties=FP_CN_GRP, color=gc,
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                          edgecolor=gc, linewidth=1.5, alpha=0.95))

    ax.set_xticks(positions)
    ax.set_xticklabels(all_labels, fontproperties=FP_CN, linespacing=1.4)

    ax.set_ylabel('Top-30% 累计总收益（元）', fontproperties=FP_CN_TITLE)
    # ax.set_title('图 12　全部模型 Top-30% 累计总收益横向对比',
    #              fontproperties=FP_CN_TITLE, pad=16)

    ax.set_ylim(y_bottom, 730000)
    ax.yaxis.set_major_formatter(
        mpl.ticker.FuncFormatter(lambda v, _: f'{v/1e4:.0f}万'))

    ax.yaxis.grid(True, linestyle=':', alpha=0.5)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # 图例：放在图外底部，水平排列，不与任何元素重叠
    from matplotlib.patches import Patch
    legend_items = [
        Patch(facecolor=COLOR_BASE,  label='基准模型'),
        Patch(facecolor=COLOR_A,     label='消融实验 A'),
        Patch(facecolor=COLOR_B,     label='消融实验 B'),
        Patch(facecolor=COLOR_C,     label='消融实验 C（完整框架）'),
        Patch(facecolor=COLOR_TREE1, label='LightGBM'),
        Patch(facecolor=COLOR_TREE2, label='XGBoost'),
        Patch(facecolor=COLOR_HYB1,  label='编码器 + LightGBM'),
        Patch(facecolor=COLOR_HYB2,  label='编码器 + XGBoost'),
    ]
    ax.legend(handles=legend_items,
              loc='upper center',
              bbox_to_anchor=(0.5, -0.1),   # 轴坐标系，负值在轴下方
              ncol=4, fontsize=10,
              framealpha=0.95, edgecolor='#CCCCCC',
              handlelength=1.2, handletextpad=0.5,
              columnspacing=1.2,
              prop=FP_CN_SM)

    # 底部留出足够空间：x轴标签 + 图例两行
    fig.subplots_adjust(bottom=0.25, top=0.91, left=0.08, right=0.95)
    path = os.path.join(OUTPUT_DIR, '图12_全部模型横向对比.png')
    fig.savefig(path, bbox_inches='tight')
    print(f'✓ 图12 已保存: {path}')
    plt.close(fig)


# ============================================================
if __name__ == '__main__':
    plot_fig11()
    plot_fig12()
    print('\n✓ 全部图表生成完毕')
